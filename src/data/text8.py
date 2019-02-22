import base64
import hashlib
import json
import shutil
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from src.logger import get_logger

logger = get_logger(__name__)


def download_data(url="http://mattmahoney.net/dc/text8.zip", dest_dir="data"):
    # prepare destination
    dest = Path(dest_dir) / Path(url).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    # downlaod zip
    if not dest.exists():
        logger.info("downloading file: %s.", url)
        r = requests.get(url, stream=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(r.raw, f)
        logger.info("file downloaded: %s.", dest)

    # extract zip
    if not Path(dest_dir, "text8").exists():
        with dest.open("rb") as f, ZipFile(f, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("file extracted.")


def load_data(src_dir="data"):
    file_path = Path(src_dir, "text8")
    with open(file_path) as f:
        text8 = f.read()
    logger.info("file loaded: %s.", file_path)
    return text8


def process_data(text8, vocab_size=None, coverage=0.9, context_size=5):
    text8_tokens = text8.split()

    # create vocab
    id2token = create_vocabulary(text8_tokens, vocab_size, coverage)
    token2id = {token: i for i, token in enumerate(id2token)}
    logger.info("vocab created, size: %s.", len(id2token))

    # compute interaction
    interaction = create_interaction_dataframe(text8_tokens, token2id, context_size)
    df = create_glove_dataframe(interaction, id2token)

    return {"vocabulary": id2token, "interaction": df}


def create_vocabulary(text_tokens, vocab_size=None, coverage=0.9):
    tokens_counter = Counter(text_tokens)
    # find cumulative proportion of token counts
    counts = np.sort(list(tokens_counter.values()))[::-1]
    counts_cumprop = np.cumsum(counts) / np.sum(counts)
    # get count with defined coverage of total tokens
    count_cutoff = counts[np.searchsorted(counts_cumprop, coverage)]
    logger.info("count cufoff: %s; token coverage: %s.", count_cutoff, coverage)
    id2token = ["<UNK>"] + [token for token, count in tokens_counter.most_common(vocab_size) if count >= count_cutoff]
    return id2token


def create_interaction_dataframe(text_tokens, token2id, context_size=5):
    token_ids = [token2id.get(token, 0) for token in text_tokens]
    df = pd.DataFrame(list(enumerate(token_ids)), columns=["position", "token_id"])
    # cross join by position for right context only
    df_concat = pd.concat([df.set_index(df["position"] + i + 1) for i in range(context_size)])
    df_co = df_concat.join(df, how="inner", lsuffix="_row", rsuffix="_column")
    df_co = df_co.loc[(df_co["token_id_row"] != df_co["token_id_column"]) &
                      (df_co["position_row"] < df_co["position_column"]), :]
    df_co = df_co.assign(**{"value": 1 / (df_co["position_column"] - df_co["position_row"])})
    # aggregate interactions
    df_agg = (df_co
              .groupby(["token_id_row", "token_id_column"])["value"]
              .agg(["count", "sum"])
              .reset_index()
              .rename(columns={"token_id_row": "row_token_id", "token_id_column": "column_token_id", "sum": "value"}))
    df_agg = df_agg.loc[(df_agg["count"] != 0) & (df_agg["value"] != 0), :]
    # union swap row and col since symmetric
    df_agg = (pd.concat([df_agg,
                         df_agg.rename(columns={"row_token_id": "column_token_id", "column_token_id": "row_token_id"})],
                        sort=False)
              .groupby(["row_token_id", "column_token_id"])
              .sum()
              .reset_index())
    logger.info("interaction dataframe created.")
    logger.info("dataframe shape: %s.", df_agg.shape)
    return df_agg


def create_interaction_matrix(text_tokens, token2id, context_size=5):
    tokens_total = len(text_tokens)
    token_ids = [token2id.get(token, 0) for token in text_tokens]
    vocab_size = len(token2id)
    interaction = sparse.dok_matrix((vocab_size, vocab_size), dtype=float)
    for i, center_id in enumerate(token_ids):
        context_ids = np.array(token_ids[max(0, i - context_size):i])
        context_len = len(context_ids)

        # count for left context and add transpose later since symmetric
        for left_i, left_id in enumerate(context_ids):
            if center_id != left_id:
                weight = 1.0 / (context_len - left_i)
                interaction[center_id, left_id] += weight

        if i % 10000 == 0:
            logger.info("%s / %s tokens processed.", i, tokens_total)
            logger.info("interaction matrix size: %s.", interaction.nnz)

    interaction = interaction.tocsr()
    interaction += interaction.transpose()
    interaction = interaction.tocoo()
    interaction.setdiag(0)
    interaction.eliminate_zeros()
    logger.info("interaction matrix computed, size: %s.", interaction.nnz)
    return interaction


def create_glove_dataframe(df, id2token, count_minimum=10):
    # apply glove transformation
    df = df.loc[df["count"] >= count_minimum, :]
    df = (df
          .assign(**{
        "row_token": np.array(id2token)[df["row_token_id"]],
        "column_token": np.array(id2token)[df["column_token_id"]],
        "glove_weight": glove_weight(df["count"]),
        "glove_value": np.log(df["value"])
    }))
    # randomise dataframe
    df = (df.set_index(df["row_token"]
                       .str.cat(df["column_token"], sep=" ")
                       .str.encode("utf8")
                       .apply(hash))
          .sort_index())
    logger.info("dataframe shape: %s.", df.shape)
    return df


def glove_weight(values, alpha=0.75, x_max=100):
    return np.clip(np.power(values / x_max, alpha), 0, 1)


def save_data(data, save_dir="data"):
    # save vocab
    vocab = data["vocabulary"]
    file_path = Path(save_dir, "vocab.json")
    with open(file_path, "w") as f:
        json.dump(vocab, f)
    logger.info("vocabulary saved: %s.", file_path)

    file_path = Path(save_dir, "vocab.txt")
    with open(file_path, "w") as f:
        f.write("\n".join(vocab))
    logger.info("vocabulary saved: %s.", file_path)

    # save interaction
    df = data["interaction"]
    file_path = Path(save_dir, "interaction.csv")
    df.to_csv(file_path, index=False)
    logger.info("interaction saved: %s.", file_path)

    return data


if __name__ == "__main__":
    parser = ArgumentParser(description="Download, extract and prepare text8 data.")
    parser.add_argument("--url", default="http://mattmahoney.net/dc/text8.zip",
                        help="url of text8 data (default: %(default)s)")
    parser.add_argument("--dest", default="data",
                        help="destination directory for downloaded and extracted files (default: %(default)s)")
    parser.add_argument("--vocab-size", default=None,
                        help="maximum size of vocab (default: %(default)s)")
    parser.add_argument("--coverage", default=0.9,
                        help="token coverage to set token count cutoff (default: %(default)s)")
    parser.add_argument("--context-size", default=5,
                        help="size of context window (default: %(default)s)")
    parser.add_argument("--log-path", default="main.log",
                        help="path of log file (default: %(default)s)")
    args = parser.parse_args()

    logger = get_logger(__name__, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s.", args)

    try:
        download_data(args.url, args.dest)
        text8 = load_data(args.dest)
        data = process_data(text8, args.vocab_size, args.coverage, args.context_size)
        save_data(data, args.dest)
    except Exception as e:
        logger.exception(e)
        raise e

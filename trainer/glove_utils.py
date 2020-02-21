import json
import os
from argparse import ArgumentParser
from datetime import datetime

import tensorflow as tf

from trainer.config import (
    BATCH_SIZE, CONFIG, EMBEDDING_SIZE, FEATURE_NAMES, L2_REG, LEARNING_RATE, OPTIMIZER, STEPS_PER_EPOCH, TOP_K,
    TRAIN_CSV, TRAIN_STEPS, VOCAB_TXT,
)
from trainer.data_utils import get_csv_dataset
from trainer.model_utils import MatrixFactorisation, get_named_variables
from trainer.utils import cosine_similarity


def build_glove_model(vocab_size, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG):
    # init layers
    mf_layer = MatrixFactorisation(vocab_size, embedding_size, l2_reg, name="glove_value")

    # build model
    inputs = [tf.keras.Input((), name=name) for name in FEATURE_NAMES]
    glove_value = mf_layer(inputs)
    glove_model = tf.keras.Model(inputs, glove_value, name="glove_model")
    return glove_model


def get_string_id_table(vocab_txt=VOCAB_TXT):
    lookup_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        vocab_txt,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
    ), 0, name="string_id_table")
    return lookup_table


def get_id_string_table(vocab_txt=VOCAB_TXT):
    lookup_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        vocab_txt,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
    ), "<UNK>", name="id_string_table")
    return lookup_table


def get_glove_dataset(file_pattern=TRAIN_CSV, vocab_txt=VOCAB_TXT, batch_size=BATCH_SIZE, num_epochs=1):
    string_id_table = get_string_id_table(vocab_txt)

    def lookup(features, targets, weights):
        features = {name: string_id_table.lookup(features[name], name=name + "_lookup") for name in FEATURE_NAMES}
        return features, targets, weights

    dataset_args = {
        "file_pattern": file_pattern,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        **CONFIG["dataset_args"],
    }
    dataset = get_csv_dataset(**dataset_args).map(lookup, num_parallel_calls=-1)
    return dataset


def get_similarity(inputs, model, vocab_txt=VOCAB_TXT, top_k=TOP_K):
    # variables
    variables = get_named_variables(model)
    embedding_layer = variables["row_embedding_layer"]
    embeddings = embedding_layer.weights[0]
    # [vocab_size, embedding_size]

    # values
    token_id = inputs[0]
    # [None]
    embed = embedding_layer(token_id)
    # [None, embedding_size]
    cosine_sim = cosine_similarity(embed, embeddings)
    # [None, vocab_size]
    top_k_sim, top_k_idx = tf.math.top_k(cosine_sim, k=top_k, name="top_k_sim")
    # [None, top_k], [None, top_k]
    id_string_table = get_id_string_table(vocab_txt)
    top_k_string = id_string_table.lookup(tf.cast(top_k_idx, tf.int64), name="string_lookup")
    # [None, top_k]
    values = {
        "embed:": embed,
        "top_k_similarity": top_k_sim,
        "top_k_string": top_k_string,
    }
    return values


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default=TRAIN_CSV,
        help="path to the training csv data (default: %(default)s)"
    )
    parser.add_argument(
        "--vocab-txt",
        default=VOCAB_TXT,
        help="path to the vocab txt (default: %(default)s)"
    )
    parser.add_argument(
        "--job-dir",
        default="checkpoints/glove",
        help="job directory (default: %(default)s)"
    )
    parser.add_argument(
        "--use-job-dir-path",
        action="store_true",
        help="flag whether to use raw job_dir path (default: %(default)s)"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=EMBEDDING_SIZE,
        help="embedding size (default: %(default)s)"
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=L2_REG,
        help="scale of l2 regularisation (default: %(default)s)"
    )
    parser.add_argument(
        "--optimizer",
        default=OPTIMIZER,
        help="name of optimzer (default: %(default)s)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="learning rate (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=TRAIN_STEPS,
        help="number of training steps (default: %(default)s)"
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=STEPS_PER_EPOCH,
        help="number of steps per checkpoint (default: %(default)s)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="number of similar items (default: %(default)s)"
    )
    args = parser.parse_args()
    params = args.__dict__

    # job_dir
    if not params["use_job_dir_path"]:
        datetime_now = datetime.now()
        job_dir = "{job_dir}_{datetime:%Y%m%d_%H%M%S}".format(job_dir=params["job_dir"], datetime=datetime_now)
        params["job_dir"] = job_dir
    tf.io.gfile.makedirs(job_dir)

    # vocab_txt
    output_vocab_txt = os.path.join(job_dir, os.path.basename(params["vocab_txt"]))
    tf.io.gfile.copy(params["vocab_txt"], output_vocab_txt, overwrite=True)
    params["vocab_txt"] = output_vocab_txt

    # save params
    params_json = os.path.join(job_dir, "params.json")
    with tf.io.gfile.GFile(params_json, "w") as f:
        json.dump(params, f, indent=2)
    return params

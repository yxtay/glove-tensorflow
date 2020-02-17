import json
import os
import sys
from argparse import ArgumentParser

import tensorflow as tf

from trainer.logger import get_logger
from trainer.glove import get_estimator, get_predict_input_fn

logger = get_logger(__name__)


def format_predictions(predictions):
    embeddings = {}
    for instance in predictions:
        # need to convert byte to string
        # and remove numpy types
        item_id = instance["row_id"].decode()
        row_embed = instance["row_embed"].tolist()
        col_embed = instance["col_embed"].tolist()

        # add entry only if valid predictions
        if item_id != "<UNK>":
            instance_embedding = {"item_id": item_id, "row_embed": row_embed, "col_embed": col_embed}
            embeddings[item_id] = instance_embedding
    logger.info("embedding dict size: %s.", len(embeddings))
    return embeddings


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--job-dir",
        default="checkpoints/glove",
        help="job directory (default: %(default)s)"
    )
    parser.add_argument(
        "--embeddings-json",
        default="checkpoints/embeddings/embeddings.json",
        help="path to the embeddings json (default: %(default)s)"
    )
    args = parser.parse_args()
    logger.info("call: %s.", " ".join(sys.argv))
    logger.info("ArgumentParser: %s.", args)

    try:
        job_dir = args.job_dir
        embeddings_json = args.embeddings_json

        # load params
        params_json = os.path.join(job_dir, "params.json")
        with tf.io.gfile.GFile(params_json) as f:
            params = json.load(f, parse_float=False)
        logger.info("params loaded: %s.", params_json)

        # estimator
        estimator = get_estimator(job_dir, params)
        logger.info("loading checkpoint: %s.", job_dir)

        # get predictions
        predict_input_fn = get_predict_input_fn(params["mappings"][params["field_names"]["row_id"]])
        predictions = estimator.predict(predict_input_fn)
        embeddings = format_predictions(predictions)

        # save predictions
        with tf.io.gfile.GFile(embeddings_json, "w") as f:
            json.dump(embeddings, f, indent=2)
        logger.info("embeddings_json saved: %s.", embeddings_json)

    except Exception as e:
        logger.exception(e)
        raise e

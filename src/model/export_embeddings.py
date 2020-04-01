import json
import os
import sys
from argparse import ArgumentParser

import tensorflow as tf

from src.logger import get_logger
from trainer.estimator import estimator_predict

logger = get_logger(__name__)


def load_json(json_path):
    with tf.io.gfile.GFile(json_path) as f:
        json_obj = json.load(f)
    logger.info("json loaded: %s.", json_path)
    return json_obj


def save_json(json_obj, json_path):
    with tf.io.gfile.GFile(json_path, "w") as f:
        json.dump(json_obj, f, indent=2)
    logger.info("json saved: %s.", json_path)
    return json_path


def format_predictions(predictions):
    embeddings = {}
    for instance in predictions:
        # need to convert byte to string
        # and remove numpy types
        item_id = instance["input_string"].decode()
        item_embedding = instance["input_embedding"].tolist()

        # add entry only if valid predictions
        if item_id != "<UNK>":
            instance_embedding = {"item_id": item_id, "item_embedding": item_embedding}
            embeddings[item_id] = instance_embedding
    logger.info("embedding dict size: %s.", len(embeddings))
    return embeddings


def main(job_dir, embeddings_json):
    # load params
    params_json = os.path.join(job_dir, "params.json")
    params = load_json(params_json)

    # get predictions
    predictions = estimator_predict(params)
    embeddings = format_predictions(predictions)

    # save predictions
    save_json(embeddings, embeddings_json)


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
        main(**args.__dict__)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(e)
        raise e

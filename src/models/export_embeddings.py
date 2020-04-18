import os
import sys
from argparse import ArgumentParser

from src.config import EMBEDDINGS_JSON, JOB_DIR
from src.models.estimator import estimator_predict
from src.utils.io import load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


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


def main(job_dir=JOB_DIR, embeddings_json=EMBEDDINGS_JSON, **kwargs):
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
        default=JOB_DIR,
        help="job directory (default: %(default)s)"
    )
    parser.add_argument(
        "--embeddings-json",
        default=EMBEDDINGS_JSON,
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

import json

import tensorflow as tf

from src.utils.logger import get_logger

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

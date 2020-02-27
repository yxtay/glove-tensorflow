# configs
import json
import os
from argparse import ArgumentParser
from datetime import datetime

import tensorflow as tf

TRAIN_CSV = "data/interaction.csv"
VOCAB_TXT = "data/vocab.txt"
EMBEDDING_SIZE = 64
L2_REG = 0.01
NEG_FACTOR = 1.
OPTIMIZER = "Adam"
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
TRAIN_STEPS = 16384
STEPS_PER_EPOCH = 16384
TOP_K = 20

# field_names
ROW_NAME = "row_token"
COL_NAME = "col_token"
TARGET_NAME = "glove_value"
WEIGHT_NAME = "glove_weight"
POS_NAME = "value"
NEG_NAME = "neg_weight"


def get_function_args(params):
    row_name = params["row_name"]
    col_name = params["col_name"]
    target_name = params["target_name"]
    weight_name = params["weight_name"]
    args = {
        "dataset_args": {
            "vocab_txt": params["vocab_txt"],
            "file_pattern": params["train_csv"],
            "batch_size": params["batch_size"],
            "feature_names": [row_name, col_name],
            "target_names": [target_name],
            "weight_name": weight_name,
        },
        "input_fn_args": {
            "file_pattern": params["train_csv"],
            "batch_size": params["batch_size"],
            "select_columns": [row_name, col_name, weight_name, target_name],
            "target_names": [target_name],
        },
        "serving_input_fn_args": {
            "string_features": [row_name, col_name],
        }
    }
    return args


def save_params(params, params_json="params.json"):
    # save params
    params_json = os.path.join(params["job_dir"], params_json)
    with tf.io.gfile.GFile(params_json, "w") as f:
        json.dump(params, f, indent=2)


def init_params(params):
    # get function args
    params.update(get_function_args(params))

    # job_dir
    if not params["disable_datetime_path"]:
        datetime_now = datetime.now()
        job_dir = "{job_dir}_{datetime:%Y%m%d_%H%M%S}".format(job_dir=params["job_dir"], datetime=datetime_now)
        params["job_dir"] = job_dir
    tf.io.gfile.makedirs(params["job_dir"])

    # vocab_txt
    output_vocab_txt = os.path.join(job_dir, os.path.basename(params["vocab_txt"]))
    tf.io.gfile.copy(params["vocab_txt"], output_vocab_txt, overwrite=True)
    params["vocab_txt"] = output_vocab_txt

    # save params
    save_params(params)
    return params


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
        "--row-name",
        default=ROW_NAME,
        help="row id name (default: %(default)s)"
    )
    parser.add_argument(
        "--col-name",
        default=COL_NAME,
        help="column id name (default: %(default)s)"
    )
    parser.add_argument(
        "--target-name",
        default=TARGET_NAME,
        help="target name (default: %(default)s)"
    )
    parser.add_argument(
        "--weight-name",
        default=WEIGHT_NAME,
        help="weight name (default: %(default)s)"
    )
    parser.add_argument(
        "--pos-name",
        default=POS_NAME,
        help="positive name (default: %(default)s)"
    )
    parser.add_argument(
        "--neg-name",
        default=NEG_NAME,
        help="negative name (default: %(default)s)"
    )
    parser.add_argument(
        "--job-dir",
        default="checkpoints/glove",
        help="job directory (default: %(default)s)"
    )
    parser.add_argument(
        "--disable-datetime-path",
        action="store_true",
        help="flag whether to disable appending datetime in job_dir path (default: %(default)s)"
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
        "--neg-factor",
        type=float,
        default=NEG_FACTOR,
        help="negative loss factor (default: %(default)s)"
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
    params = init_params(args.__dict__)
    return params

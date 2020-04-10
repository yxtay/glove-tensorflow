import os
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path


def read_config(ini_file="app.ini", environment=os.environ.get("ENVIRONMENT", "dev")):
    # read configs
    ini_path = Path("configs", ini_file)
    parser = ConfigParser()
    parser.read([ini_path])
    # environment config
    config = parser[environment]
    return config


CONFIG = read_config()

# paths
JOB_DIR = CONFIG["JOB_DIR"]

# files
TRAIN_CSV = CONFIG["TRAIN_CSV"]
VOCAB_TXT = CONFIG["VOCAB_TXT"]
EMBEDDINGS_JSON = CONFIG["EMBEDDINGS_JSON"]

# preprocess
TEXT8_URL = CONFIG["TEXT8_URL"]
DATA_DIR = CONFIG["DATA_DIR"]
VOCAB_SIZE = None
COVERAGE = CONFIG.getfloat("COVERAGE")
CONTEXT_SIZE = CONFIG.getint("CONTEXT_SIZE")

# data
ROW_NAME = CONFIG["ROW_NAME"]
COL_NAME = CONFIG["COL_NAME"]
TARGET_NAME = CONFIG["TARGET_NAME"]
WEIGHT_NAME = CONFIG["WEIGHT_NAME"]
POS_NAME = CONFIG["POS_NAME"]
NEG_NAME = CONFIG["NEG_NAME"]
STRING_IDX = CONFIG.getint("STRING_IDX")
NAME_IDX = CONFIG.getint("NAME_IDX")

# model
EMBEDDING_SIZE = CONFIG.getint("EMBEDDING_SIZE")
L2_REG = CONFIG.getfloat("L2_REG")
NEG_FACTOR = CONFIG.getfloat("NEG_FACTOR")
OPTIMIZER = CONFIG["OPTIMIZER"]
LEARNING_RATE = CONFIG["LEARNING_RATE"]
BATCH_SIZE = CONFIG.getint("BATCH_SIZE")
TRAIN_STEPS = CONFIG.getint("TRAIN_STEPS")
STEPS_PER_EPOCH = CONFIG.getint("STEPS_PER_EPOCH")
TOP_K = CONFIG.getint("TOP_K")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("key", help="key name to get value")
    args = parser.parse_args()

    key = args.key
    sys.stdout.write(CONFIG[key])
    sys.stdout.flush()

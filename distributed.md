# Distributed Model Training and Evaluation on Google Cloud Platform

The trainer module in this repository also allows for distributed model training and evaluation on Google Cloud Platform.

## Setup

```bash
# clone repo
git clone git@github.com:yxtay/glove-tensorflow.git && cd recommender-tensorflow

# create conda environment
conda env create -f=environment.yml

# activate environment
source activate dl
```

You may also use accompanying docker commands to avoid environment setup.

## Download & Process Data

```bash
python -m src.data.text8
```

**With Docker**

```bash
docker run --rm -w=/home \
  --mount type=bind,source=$(pwd),target=/home \
  continuumio/anaconda3:5.3.0 \
  python -m src.data.text8
```

**Usage**

```
usage: text8.py [-h] [--url URL] [--dest DEST] [--vocab-size VOCAB_SIZE]
                [--coverage COVERAGE] [--context-size CONTEXT_SIZE] [--reset]
                [--log-path LOG_PATH]

Download, extract and prepare text8 data.

optional arguments:
  -h, --help            show this help message and exit
  --url URL             url of text8 data (default:
                        http://mattmahoney.net/dc/text8.zip)
  --dest DEST           destination directory for downloaded and extracted
                        files (default: data)
  --vocab-size VOCAB_SIZE
                        maximum size of vocab (default: None)
  --coverage COVERAGE   token coverage to set token count cutoff (default:
                        0.9)
  --context-size CONTEXT_SIZE
                        size of context window (default: 5)
  --reset               whether to recompute interactions
  --log-path LOG_PATH   path of log file (default: main.log)
```

## Google Cloud Platform Credentials

TODO: Add instructions to set up `gcloud`.

## Upload files to Google Cloud Storage

```bash
# change accordingly
BUCKET=default-223103
DIR_PATH=glove-tensorflow

SRC=$(pwd)/data
DST=gs://${BUCKET}/${DIR_PATH}
gsutil -m cp -r "${SRC}" ${DST}
```

## Distributed Training & Evaluation

```bash
# change accordingly
RUNTIME_VERSION=1.13
PYTHON_VERSION=3.5
REGION=us-central1
SCALE_TIER=basic

BUCKET=default-223103
DIR_PATH=glove-tensorflow
PACKAGE_NAME=trainer
MODEL_NAME=glove

JOB_NAME=${MODEL_NAME}_$(date -u +%y%m%d_%H%M%S)
OUTPUT_PATH=gs://${BUCKET}/${DIR_PATH}/checkpoints/${JOB_NAME}
TRAIN_CSV=gs://${BUCKET}/${DIR_PATH}/data/interaction.csv
VOCAB_JSON=gs://${BUCKET}/${DIR_PATH}/data/vocab.json

gcloud ml-engine jobs submit training $JOB_NAME \
    --package-path ${PACKAGE_NAME} \
    --module-name ${PACKAGE_NAME}.${MODEL_NAME} \
    --job-dir ${OUTPUT_PATH} \
    --runtime-version ${RUNTIME_VERSION} \
    --python-version ${PYTHON_VERSION} \
    --region ${REGION} \
    --scale-tier ${SCALE_TIER} \
    -- \
    --train-csv ${TRAIN_CSV} \
    --vocab-json ${VOCAB_JSON} \
    --train-steps 65536
```

## Tensorboard

You may inspect model training metrics with Tensorboard

```bash
BUCKET=default-223103
DIR_PATH=glove-tensorflow
OUTPUT_PATH=gs://${BUCKET}/${DIR_PATH}/checkpoints

tensorboard --logdir OUTPUT_PATH
```

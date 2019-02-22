#/bin/bash
set -euxo pipefail

BUCKET="default-223103"
DIR_PATH="glove-tensorflow"
REGION="us-central1"
RUNTIME_VERSION=1.12

MODEL_TYPE="glove"
JOB_NAME="${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)"
OUTPUT_PATH="gs://${BUCKET}/${DIR_PATH}/checkpoints/$JOB_NAME"
PACKAGE_PATH="trainer"
TRAIN_CSV="gs://${BUCKET}/${DIR_PATH}/data/interaction.csv"
VOCAB_JSON="gs://${BUCKET}/${DIR_PATH}/data/vocab.json"

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version ${RUNTIME_VERSION} \
    --module-name ${PACKAGE_PATH}.${MODEL_TYPE} \
    --package-path ${PACKAGE_PATH} \
    --region $REGION \
    -- \
    --train-csv ${TRAIN_CSV} \
    --vocab-json ${VOCAB_JSON} \
    --train-steps 100000

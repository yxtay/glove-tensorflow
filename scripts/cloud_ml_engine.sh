#/bin/bash
set -euxo pipefail

REGION="us-central1"
RUNTIME_VERSION="1.12"

BUCKET="default-223103"
DIR_PATH="glove-tensorflow"
PACKAGE_NAME="trainer"
MODEL_NAME="glove"

JOB_NAME="${MODEL_NAME}_$(date -u +%y%m%d_%H%M%S)"
OUTPUT_PATH="gs://${BUCKET}/${DIR_PATH}/checkpoints/${JOB_NAME}"
TRAIN_CSV="gs://${BUCKET}/${DIR_PATH}/data/interaction.csv"
VOCAB_JSON="gs://${BUCKET}/${DIR_PATH}/data/vocab.json"

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --job-dir ${OUTPUT_PATH} \
    --runtime-version ${RUNTIME_VERSION} \
    --module-name ${PACKAGE_NAME}.${MODEL_NAME} \
    --package-path ${PACKAGE_NAME} \
    --region $REGION \
    -- \
    --train-csv ${TRAIN_CSV} \
    --vocab-json ${VOCAB_JSON} \
    --train-steps 100000

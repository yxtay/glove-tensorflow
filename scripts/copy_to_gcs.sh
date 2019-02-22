#/bin/bash
set -euxo pipefail

BUCKET="default-223103"
DIR_PATH="glove-tensorflow"

SRC=$(pwd)/data/*
DST=gs://${BUCKET}/${DIR_PATH}/data/
gsutil -m cp -r "${SRC}" ${DST}

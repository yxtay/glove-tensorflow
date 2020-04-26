# Distributed Model Training and Evaluation on Google Cloud Platform

The trainer module in this repository also allows for distributed model training and evaluation on Google Cloud Platform.

## Setup

```bash
ENV_NAME=glove-tensorflow

# clone repo
git clone git@github.com:yxtay/glove-tensorflow.git && cd recommender-tensorflow

# create and activate conda environment
conda env create -n ${ENV_NAME} -y python=3.7
conda activate ${ENV_NAME}

# install requirements
# make install-requirments
pip install -r requirements/main.txt -r requirements/dev.txt
```

You may also use accompanying docker commands to avoid environment setup.

## Download & Process Data

```bash
python -m src.data.text8
```

**With Docker**

```bash
# make docker-data
docker run --rm -w=/home \
  --mount type=bind,source=$(pwd),target=/home \
  continuumio/anaconda3:2019.10 \
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

## Upload files to Google Cloud Storage

Upload processed data files onto Google Cloud Storage.
This is required for the model training on AI Platform to be able to access and download the files.

```bash
# make up-data
APP_NAME = glove-tensorflow
DATA_DIR = data
GCS_BUCKET = <gcs_bucket>
GCS_BUCKET_PATH = gs://$(GCS_BUCKET)/$(APP_NAME)
gsutil -m rsync -r $(DATA_DIR) $(GCS_BUCKET_PATH)/$(DATA_DIR)
```

## AI Platform Training & Evaluation

Training the model with AI Platform using custom containers.

```bash
# make train-ai-platform
APP_NAME = glove-tensorflow
ENVIRONMENT = dev
JOB_NAME = $(APP_NAME)
GCP_PRODJECT = <gcp_project>
FULL_IMAGE_NAME = gcr.io/$(GCP_PRODJECT)/$(APP_NAME)
IMAGE_TAG = latest
GCS_BUCKET = <gcs_bucket>
CHECKPOINT_DIR = checkpoints
GCS_JOB_DIR = gs://$(GCS_BUCKET)/$(APP_NAME)/$(CHECKPOINT_DIR)/$(JOB_NAME)

gcloud ai-platform jobs submit training $(subst -,_,$(JOB_NAME)) \
    --labels app=$(APP_NAME),environment=$(ENVIRONMENT) \
    --project $(GCP_PROJECT) \
    --region us-central1 \
    --scale-tier BASIC \
    --master-image-uri $(FULL_IMAGE_NAME):$(IMAGE_TAG) \
    --stream-logs \
    -- \
    make train-loop \
    JOB_DIR=$(GCS_JOB_DIR) \
    2>&1
```

## Tensorboard

You may inspect model training metrics with Tensorboard

```bash
JOB_NAME = glove-tensorflow
GCS_BUCKET = <gcs_bucket>
CHECKPOINT_DIR = checkpoints
GCS_JOB_DIR = gs://$(GCS_BUCKET)/$(APP_NAME)/$(CHECKPOINT_DIR)/$(JOB_NAME)

tensorboard --logdir $(GCS_JOB_DIR)
```

## Deploy on Cloud Run

```bash
# make cloud-run-deploy
APP_NAME = glove-tensorflow
ENVIRONMENT = dev
GCP_PRODJECT = <gcp_project>
FULL_SERVING_IMAGE_NAME = gcr.io/$(GCP_PRODJECT)/$(APP_NAME)-tfserving
IMAGE_TAG = latest
GCS_BUCKET = <gcs_bucket>
EXPORT_DIR = export
GCS_EXPORT_PATH = gs://$(GCS_BUCKET)/$(APP_NAME)/$(EXPORT_DIR)

gcloud run deploy $(APP_NAME) \
    --project $(GCP_PROJECT) \
    --region us-central1 \
    --platform managed \
    --labels app=$(APP_NAME),environment=$(ENVIRONMENT) \
    --image $(FULL_SERVING_IMAGE_NAME):$(IMAGE_TAG) \
    --set-env-vars=GCS_EXPORT_PATH=$(GCS_EXPORT_PATH) \
    --memory 512Mi \
    --no-allow-unauthenticated \
    2>&1
```

This command returns the service url required for querying the model in the next step.

## Query Cloud Run

```bash
# make cloud-run-query
APP_NAME = glove-tensorflow
SERVICE_URL = <service_url>
curl \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -X POST \
  $(SERVICE_URL)/v1/models/$(APP_NAME):predict \
  -d '{"instances": [{"row_token": "man", "col_token": "man"}]}'
```


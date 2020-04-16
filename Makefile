MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SUFFIXES:

DOCKER_FILE := Dockerfile
SERVING_DOCKER_FILE := serving.Dockerfile

# project
ENVIRONMENT ?= dev
ARGS =
GOOGLE_APPLICATION_CREDENTIALS ?=
APP_NAME = $(shell python -m src.config APP_NAME)
MODEL_NAME = $(shell python -m src.config MODEL_NAME)

# gcp
GCP_PROJECT = $(shell python -m src.config GCP_PROJECT)
GCS_BUCKET = $(shell python -m src.config GCS_BUCKET)
GCS_BUCKET_PATH = gs://$(GCS_BUCKET)/$(APP_NAME)
SERVICE_URL = $(shell python -m src.config SERVICE_URL)

# docker
IMAGE_HOST = $(shell python -m src.config IMAGE_HOST)
IMAGE_REPO = $(shell python -m src.config IMAGE_REPO)
FULL_IMAGE_NAME = $(IMAGE_HOST)/$(IMAGE_REPO)/$(APP_NAME)
FULL_SERVING_IMAGE_NAME = $(FULL_IMAGE_NAME)-tfserving
IMAGE_TAG ?= latest

# paths
DATA_DIR = $(shell python -m src.config DATA_DIR)
CHECKPOINTS_DIR = $(shell python -m src.config CHECKPOINTS_DIR)
EXPORT_DIR = $(shell python -m src.config EXPORT_DIR)

# train
DATETIME := $(shell date +%Y%m%d-%H%M%S)
JOB_NAME = $(subst _,-,$(MODEL_NAME)-$(DATETIME))
JOB_DIR = $(CHECKPOINTS_DIR)/$(JOB_NAME)

# gcs
GCS_JOB_DIR = $(GCS_BUCKET_PATH)/$(JOB_DIR)
GCS_MODEL_DIR = $(GCS_BUCKET_PATH)/$(CHECKPOINTS_DIR)/$(APP_NAME)
GCS_EXPORT_PATH = $(GCS_BUCKET_PATH)/$(EXPORT_DIR)

.PHONY: all
all: data train embeddings

.PHONY: data
data:
	python -m src.data.text8

.PHONY: docker-data
docker-data:
	docker run --rm -w=/home \
	  --mount type=bind,source=$(shell pwd),target=/home \
	  continuumio/anaconda3:2019.10 \
	  python -m src.data.text8

.PHONY: train
train:
	python -m src.models.$(MODEL_NAME) \
		--job-dir $(JOB_DIR) \
		--disable-datetime-path \
		$(ARGS)

.PHONY: docker-train
docker-train:
	docker run --rm -w=/home \
	  --mount type=bind,source=$(shell pwd),target=/home \
	  tensorflow/tensorflow:2.1.0-py3 \
	  python -m src.models.$(MODEL_NAME) \
	  --job-dir $(JOB_DIR) \
	  --disable-datetime-path \
	  $(ARGS)

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir $(CHECKPOINTS_DIR)

.PHONY: docker-tensorboard
docker-tensorboard:
	docker run --rm -w=/home -p 6006:6006 \
	  --mount type=bind,source=$(shell pwd),target=/home \
	  tensorflow/tensorflow:2.1.0-py3 \
	  tensorboard --logdir $(CHECKPOINTS_DIR)

.PHONY: saved-model-cli
saved-model-cli:
	saved_model_cli show --all --dir $(JOB_DIR)

.PHONY: serving
serving:
	docker run --rm -p 8500:8500 -p 8501:8501 \
	  --mount type=bind,source=$(shell pwd)/$(JOB_DIR)/export/exporter,target=/models/$(MODEL_NAME) \
	  -e MODEL_NAME=$(MODEL_NAME) \
	  tensorflow/serving:2.1.0

.PHONY: query
query:
	curl -X POST \
	  http://localhost:8501/v1/models/$(MODEL_NAME):predict \
	  -d '{"instances": [{"row_token": "man", "col_token": "man"}]}'

.PHONY: embeddings
embeddings:
	python -m src.models.export_embeddings --job-dir $(JOB_DIR)

.PHONY: up-data
up-data:
	gsutil -m rsync -r $(DATA_DIR) $(GCS_BUCKET_PATH)/$(DATA_DIR)

.PHONY: dl-data
dl-data:
	mkdir -p $(DATA_DIR)
	gsutil -m rsync -r $(GCS_BUCKET_PATH)/$(DATA_DIR) $(DATA_DIR)

.PHONY: train-loop
train-loop: dl-data train
	gsutil -m rsync -dr $(JOB_DIR) $(GCS_MODEL_DIR) 2>&1
	gsutil -m rsync -dr $(GCS_MODEL_DIR)/export/exporter $(GCS_EXPORT_PATH) 2>&1

.PHONY: train-ai-platform
train-ai-platform:
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
		"ARGS=$(ARGS)" \
		2>&1

.PHONY: dl-checkpoints
dl-checkpoints:
	mkdir -p $(CHECKPOINTS_DIR)
	gsutil -m rsync -r $(GCS_BUCKET_PATH)/$(CHECKPOINTS_DIR) $(CHECKPOINTS_DIR)

.PHONY: dl-export
dl-export:
	mkdir -p $(EXPORT_DIR)
	gsutil -m rsync -r $(GCS_EXPORT_PATH) $(EXPORT_DIR)

.PHONY: update-requirements
update-requirements:
	pip install --upgrade pip setuptools pip-tools
	pip-compile --upgrade --build-isolation --output-file requirements/main.txt requirements/main.in
	pip-compile --upgrade --build-isolation --output-file requirements/dev.txt requirements/dev.in

.PHONY: install-requirements
install-requirements:
	pip install -r requirements/main.txt -r requirements/dev.txt

.PHONY: sync-requirements
sync-requirements:
	pip-sync requirements/main.txt requirements/dev.txt

.PHONY: docker-build
docker-build:
	docker pull $(FULL_IMAGE_NAME):$(IMAGE_TAG) || exit 0
	docker build \
	  --build-arg ENVIRONMENT=$(ENVIRONMENT) \
	  --cache-from $(FULL_IMAGE_NAME):$(IMAGE_TAG) \
	  --tag $(FULL_IMAGE_NAME):$(IMAGE_TAG) \
	  --file $(DOCKER_FILE) .

.PHONY: docker-push
docker-push:
	docker push $(FULL_IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-run
docker-run:
	docker run --rm -it \
	  --mount type=bind,source=$(shell pwd)/secrets,target=/app/secrets \
	  -e GOOGLE_APPLICATION_CREDENTIALS=$(GOOGLE_APPLICATION_CREDENTIALS) \
	  -e ENVIRONMENT=$(ENVIRONMENT) \
	  $(FULL_IMAGE_NAME):$(IMAGE_TAG) \
	  $(ARGS)

.PHONY: docker-exec
docker-exec:
	docker exec -it \
	  $(shell docker ps -q  --filter ancestor=$(FULL_IMAGE_NAME):$(IMAGE_TAG)) \
	  /bin/bash

.PHONY: docker-stop
docker-stop:
	docker stop \
	  $(shell docker ps -q  --filter ancestor=$(FULL_IMAGE_NAME):$(IMAGE_TAG))

.PHONY: docker-build-serving
docker-build-serving:
	docker pull $(FULL_SERVING_IMAGE_NAME):$(IMAGE_TAG) || exit 0
	docker build \
	  --cache-from $(FULL_SERVING_IMAGE_NAME):$(IMAGE_TAG) \
	  --tag $(FULL_SERVING_IMAGE_NAME):$(IMAGE_TAG) \
	  --file $(SERVING_DOCKER_FILE) .

.PHONY: docker-push-serving
docker-push-serving:
	docker push $(FULL_SERVING_IMAGE_NAME):$(IMAGE_TAG)

.PHONY: cloud-run-deploy
cloud-run-deploy:
	gcloud run deploy ${APP_NAME} \
		--project $(GCP_PROJECT) \
		--region us-central1 \
		--platform managed \
		--labels app=$(APP_NAME),environment=$(ENVIRONMENT) \
		--image $(FULL_SERVING_IMAGE_NAME):$(IMAGE_TAG) \
		--set-env-vars=GCS_EXPORT_PATH=$(GCS_EXPORT_PATH) \
		--memory 512Mi \
		--no-allow-unauthenticated \
		2>&1

.PHONY: cloud-run-query
cloud-run-query:
	curl \
	  -H "Authorization: Bearer $(shell gcloud auth print-identity-token)" \
	  -X POST \
	  $(SERVICE_URL)/v1/models/$(APP_NAME):predict \
	  -d '{"instances": [{"row_token": "man", "col_token": "man"}]}'

.PHONY: update-requirements
update-requirements:
	pip install --upgrade pip setuptools pip-tools
	pip-compile --upgrade --build-isolation --output-file requirements/main.txt requirements/main.in
	pip-compile --upgrade --build-isolation --output-file requirements/dev.txt requirements/dev.in

.PHONY: install-requirements
install-requirements:
	pip install -r requirements/main.txt -r requirements/dev.txt

.PHONY: data
data:
	python -m src.data.text8

.PHONY: docker-data
docker-data:
	docker run --rm -w=/home \
	  --mount type=bind,source=$(pwd),target=/home \
	  continuumio/anaconda3:2019.10 \
	  python -m src.data.text8

.PHONY: train-keras
train-keras:
	python -m trainer.train_keras

.PHONY: train-estimator
train-estimator:
	python -m trainer.train_estimator

.PHONY: docker-train-estimator
docker-train:
	docker run --rm -w=/home \
	  --mount type=bind,source=$(pwd),target=/home \
	  tensorflow/tensorflow:2.1.0-py3 \
	  python -m trainer.train_estimator

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir checkpoints/

.PHONY: docker-tensorboard
docker-tensorboard:
	docker run --rm -w=/home -p 6006:6006 \
	  --mount type=bind,source=$(pwd),target=/home \
	  tensorflow/tensorflow:2.1.0-py3 \
	  tensorboard --logdir checkpoints/

.PHONY: serving
serving:
	docker run --rm -p 8500:8500 -p 8501:8501 \
	  --mount type=bind,source=$(shell pwd)/checkpoints/glove/export/exporter,target=/models/glove \
	  -e MODEL_NAME=glove -t tensorflow/serving:2.1.0-py3

.PHONY: query
query:
	curl -X POST \
	  http://localhost:8501/v1/models/glove:predict \
	  -d '{"instances": [{"row_token": "man", "col_token": "man"}]}'

.PHONY: embeddings
embeddings:
	python -m src.model.export_embeddings

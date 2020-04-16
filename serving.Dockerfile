FROM tensorflow/serving:2.1.0
MAINTAINER wyextay@gmail.com

ENV MODEL_NAME=glove-tensorflow
ENV GCS_EXPORT_PATH=gs://${GCS_BUCKET}/${MODEL_NAME}/export

ENV GRPC_PORT=8500 PORT=8501
ENTRYPOINT tensorflow_model_server --port=${GRPC_PORT} --rest_api_port=${PORT} \
    --model_name=${MODEL_NAME} --model_base_path=${GCS_EXPORT_PATH}

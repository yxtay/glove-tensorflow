FROM python:3.7.6-slim AS base
MAINTAINER wyextay@gmail.com

# install google cloud sdk
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN apt-get update && apt-get install --no-install-recommends --yes curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get remove --purge --autoremove --yes curl gnupg && \
    apt-get update && apt-get install --no-install-recommends --yes google-cloud-sdk python-crcmod && \
    rm -rf /var/lib/apt/lists/*
# end of google cloud sdk install

RUN apt-get update && apt-get install --no-install-recommends --yes make && \
    rm -rf /var/lib/apt/lists/*

# set up environment
ENV HOME=/app
ENV VIRTUAL_ENV=$HOME/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
WORKDIR $HOME

# set up python
RUN python -m venv $VIRTUAL_ENV && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python --version

# install dependencies
COPY requirements.txt requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip freeze

COPY configs configs
COPY Makefile Makefile
COPY src src

ARG ENVIRONMENT=dev
ENV ENVIRONMENT $ENVIRONMENT
CMD ["make"]

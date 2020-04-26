# GloVe Model in TensorFlow

Implementation of GloVe using TensorFlow estimator API.

The trainer module in this repository also allows for cloud model training and evaluation on Google Cloud Platform. Please refer to [cloud](cloud.md).

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

The [text8 dataset](http://mattmahoney.net/dc/textdata.html) is used for demonstration purposes.
The following script downloads the data, processes it to prepare the vocabulary and cooccurrence matrix. The data is serialised to `csv`.

```bash
# make data
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

**Sample data**

|   row_token_id |   col_token_id |   count |    value | row_token   | col_token   |   neg_weight |   glove_weight |   glove_value |
|---------------:|---------------:|--------:|---------:|:------------|:------------|-------------:|---------------:|--------------:|
|           6125 |             38 |      24 |  16.9500 | altogether  | not         |    0.6421    |       0.3428   |       2.83027 |
|             18 |           1571 |     176 |  74.1000 | was         | prominent   |    7.5889    |       1.0000   |       4.30542 |
|             91 |            372 |      19 |   5.4500 | th          | society     |    3.1999    |       0.2877   |       1.69562 |
|            432 |            541 |      12 |   5.9000 | numbers     | note        |    0.6461    |       0.2038   |       1.77495 |
|           1304 |            285 |      25 |  11.1667 | na          | europe      |    0.4112    |       0.3535   |       2.41293 |
|             32 |             18 |    2312 | 723.2000 | be          | was         |  406.5180    |       1.0000   |       6.58369 |
|           2247 |           1154 |     136 |  46.5833 | html        | www         |    0.0740    |       1.0000   |       3.84124 |
|            710 |            229 |      18 |   9.0500 | cannot      | point       |    0.8569    |       0.2763   |       2.20276 |
|            467 |           3756 |      12 |   5.2000 | style       | width       |    0.0911    |       0.2038   |       1.64866 |
|             80 |            543 |      35 |  20.6333 | over        | lost        |    2.6989    |       0.4550   |       3.02691 | 

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

## Train GloVe

### Estimator

```bash
# make train
python -m trainer.estimator
```

**With Docker**

```bash
# make docker-train
docker run --rm -w=/home \
  --mount type=bind,source=$(pwd),target=/home \
  tensorflow/tensorflow:2.1.0-py3 \
  python -m trainer.estimator
```

**Usage**

```
usage: estimator.py [-h] [--train-csv TRAIN_CSV] [--vocab-txt VOCAB_TXT]
                    [--row-name ROW_NAME] [--col-name COL_NAME]
                    [--target-name TARGET_NAME] [--weight-name WEIGHT_NAME]
                    [--pos-name POS_NAME] [--neg-name NEG_NAME]
                    [--job-dir JOB_DIR] [--disable-datetime-path]
                    [--embedding-size EMBEDDING_SIZE] [--l2-reg L2_REG]
                    [--neg-factor NEG_FACTOR] [--optimizer OPTIMIZER]
                    [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE]
                    [--train-steps TRAIN_STEPS]
                    [--steps-per-epoch STEPS_PER_EPOCH] [--top-k TOP_K]

optional arguments:
  -h, --help            show this help message and exit
  --train-csv TRAIN_CSV
                        path to the training csv data (default:
                        data/interaction.csv)
  --vocab-txt VOCAB_TXT
                        path to the vocab txt (default: data/vocab.txt)
  --row-name ROW_NAME   row id name (default: row_token)
  --col-name COL_NAME   column id name (default: col_token)
  --target-name TARGET_NAME
                        target name (default: glove_value)
  --weight-name WEIGHT_NAME
                        weight name (default: glove_weight)
  --pos-name POS_NAME   positive name (default: value)
  --neg-name NEG_NAME   negative name (default: neg_weight)
  --job-dir JOB_DIR     job directory (default: checkpoints/glove)
  --disable-datetime-path
                        flag whether to disable appending datetime in job_dir
                        path (default: False)
  --embedding-size EMBEDDING_SIZE
                        embedding size (default: 64)
  --l2-reg L2_REG       scale of l2 regularisation (default: 0.01)
  --neg-factor NEG_FACTOR
                        negative loss factor (default: 1.0)
  --optimizer OPTIMIZER
                        name of optimzer (default: Adam)
  --learning-rate LEARNING_RATE
                        learning rate (default: 0.001)
  --batch-size BATCH_SIZE
                        batch size (default: 1024)
  --train-steps TRAIN_STEPS
                        number of training steps (default: 16384)
  --steps-per-epoch STEPS_PER_EPOCH
                        number of steps per checkpoint (default: 16384)
  --top-k TOP_K         number of similar items (default: 20)
```

### Keras

```bash
# make train MODEL_NAME=keras
python -m trainer.keras
```

### Logistic Matrix Factorisation

```bash
# make train MODEL_NAME=logistic_matrix_factorisation
python -m trainer.logistic_matrix_factorisation
```

## Tensorboard

You may inspect model training metrics with Tensorboard.

```bash
# make tensorboard
CHECKPOINTS_DIR=checkpoints

tensorboard --logdir ${CHECKPOINTS_DIR}
```

**With Docker**

```bash
# make docker-tensorboard
CHECKPOINTS_DIR=checkpoints

docker run --rm -w=/home -p 6006:6006 \
  --mount type=bind,source=$(pwd),target=/home \
  tensorflow/tensorflow:2.1.0-py3 \
  tensorboard --logdir ${CHECKPOINTS_DIR}
```

Access [Tensorboard](http://localhost:6006/) on your browser

## TensorFlow Serving

The trained and serialised model may be served with TensorFlow Serving.

```bash
# make serving MODEL_NAME=glove
JOB_DIR=checkpoints/glove_estimator
MODEL_NAME=glove

docker run --rm -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=$(pwd)/${JOB_DIR}/export/exporter,target=/models/${MODEL_NAME} \
  -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving:2.1.0
```

**Model signature**

```bash
# make saved-model-cli JOB_DIR=checkpoints/glove_estimator/export/exporter/1582880583
JOB_DIR=checkpoints/glove_estimator/export/exporter/1582880583

saved_model_cli show --all --dir ${JOB_DIR}
```

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['col_token'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: col_token:0
    inputs['row_token'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: row_token:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['input_embedding'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 64)
        name: predictions/row_embedding/embedding_lookup/Identity_1:0
    outputs['input_string'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: predictions/input_string_lookup/LookupTableFindV2:0
    outputs['top_k_similarity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 20)
        name: predictions/top_k_sim:0
    outputs['top_k_string'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 20)
        name: predictions/top_k_string_lookup/LookupTableFindV2:0
  Method name is: tensorflow/serving/predict
```

Once served, you may query the model with the following command.

Sample request

```bash
# make query MODEL_NAME=glove
MODEL_NAME=glove

curl -X POST \
  http://localhost:8501/v1/models/${MODEL_NAME}:predict \
  -d '{"instances": [{"row_token": "man", "col_token": "man"}]}'
```

Sample response

```
{
  "predictions": [
    {
      "input_embedding": [
        -0.39519611,
        -0.000384220504,
        0.360801637,
        0.71601522,
        -0.425830722,
        -0.259146929,
        -0.13219744,
        0.307031065,
        0.695665479,
        -0.504015446,
        ...
      ],
      "input_string": "man",
      "top_k_similarity": [
        0.99999994,
        0.725774705,
        0.707765281,
        0.693533063,
        0.679038405,
        0.646895647,
        0.642417192,
        0.640380502,
        0.63178885,
        0.631023884,
        ...
      ],
      "top_k_string": [
        "man",
        "person",
        "god",
        "woman",
        "young",
        "movie",
        "great",
        "good",
        "himself",
        "son",
        ...
      ]
    }
  ]
}
```

## Cloud

For cloud model training and evaluation, please refer to [cloud](cloud.md).

## References

- Mahoney, M. (2006). Large Text Compression Benchmark. Retrieved from [http://mattmahoney.net/dc/text.html](http://mattmahoney.net/dc/text.html).
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. \[[pdf](https://nlp.stanford.edu/pubs/glove.pdf)\]\[[bib](https://nlp.stanford.edu/pubs/glove.bib)\]

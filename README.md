# GloVe Model in TensorFlow

Implementation of GloVe using TensorFlow estimator API.

The trainer module in this repository also allows for distributed model training and evaluation on Google Cloud Platform. Please refer to [distributed](distributed.md).

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

The [text8 dataset](http://mattmahoney.net/dc/textdata.html) is used for demonstration purposes.
The following script downloads the data, processes it to prepare the vocabulary and cooccurrence matrix. The data is serialised to `csv`.

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

**Sample data**

| row_token_id | column_token_id | count | value   | row_token    | column_token | glove_weight | glove_value | 
|--------------|-----------------|-------|---------|--------------|--------------|--------------|-------------| 
| 614          | 848             | 16    | 12.6499 | irish        | origin       | 0.2529       | 2.5376      | 
| 113          | 1133            | 27    | 11.1333 | number       | places       | 0.3745       | 2.4099      | 
| 4501         | 2158            | 12    |  3.6833 | discrete     | continuous   | 0.2038       | 1.3038      | 
| 6007         | 1               | 110   | 51.5166 | videos       | the          | 1.0000       | 3.9419      | 
| 153          | 65              | 57    | 19.0500 | general      | time         | 0.6560       | 2.9470      | 
| 2978         | 642             | 12    |  6.1166 | consumer     | food         | 0.2038       | 1.8110      | 
| 2156         | 59              | 41    | 19.4000 | historically | used         | 0.5123       | 2.9652      | 
| 3166         | 45              | 23    | 16.3666 | collapse     | its          | 0.3321       | 2.7952      | 
| 445          | 100             | 32    | 14.2333 | center       | where        | 0.4254       | 2.6555      | 

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

```bash
python -m trainer.glove
```

**With Docker**

```bash
docker run --rm -w=/home \
  --mount type=bind,source=$(pwd),target=/home \
  tensorflow/tensorflow:1.12.0-py3 \
  python -m trainer.glove
```

**Usage**

```
usage: glove.py [-h] [--train-csv TRAIN_CSV] [--vocab-json VOCAB_JSON]
                [--job-dir JOB_DIR] [--restore]
                [--embedding-size EMBEDDING_SIZE] [--k K]
                [--batch-size BATCH_SIZE] [--train-steps TRAIN_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --train-csv TRAIN_CSV
                        path to the training csv data (default:
                        data/interaction.csv)
  --vocab-json VOCAB_JSON
                        path to the vocab json (default: data/vocab.json)
  --job-dir JOB_DIR     job directory (default: checkpoints/glove)
  --restore             whether to restore from JOB_DIR
  --embedding-size EMBEDDING_SIZE
                        embedding size (default: 64)
  --k K                 k for top k similarity (default: 100)
  --batch-size BATCH_SIZE
                        batch size (default: 1024)
  --train-steps TRAIN_STEPS
                        number of training steps (default: 20000)
```

## Tensorboard

You may inspect model training metrics with Tensorboard.

```bash
tensorboard --logdir checkpoints/
```

**With Docker**

```bash
docker run --rm -w=/home -p 6006:6006 \
  --mount type=bind,source=$(pwd),target=/home \
  tensorflow/tensorflow:1.12.0-py3 \
  tensorboard --logdir checkpoints/
```

Access [Tensorboard](http://localhost:6006/) on your browser

## TensorFlow Serving

The trained and serialised model may be served with TensorFlow Serving.

```bash
CHECKPOINT_PATH=checkpoints/glove/export/exporter

docker run --rm -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=$(pwd)/${CHECKPOINT_PATH},target=/models/glove \
  -e MODEL_NAME=glove -t tensorflow/serving:1.12.0
```

**Model signature**

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['column_token'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: column_token:0
    inputs['row_token'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: row_token:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['column_bias'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: mf/column_token_bias_lookup/Identity:0
    outputs['column_embed'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 64)
        name: mf/column_token_embed_lookup/Identity:0
    outputs['embed_norm_product'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: similarity/Sum:0
    outputs['row_bias'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: mf/row_token_bias_lookup/Identity:0
    outputs['row_embed'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 64)
        name: mf/row_token_embed_lookup/Identity:0
    outputs['top_k_column_similarity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: similarity/top_k_sim_column_token:0
    outputs['top_k_column_string'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 100)
        name: similarity/column_token_string_lookup_Lookup:0
    outputs['top_k_row_similarity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: similarity/top_k_sim_row_token:0
    outputs['top_k_row_string'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 100)
        name: similarity/row_token_string_lookup_Lookup:0
  Method name is: tensorflow/serving/predict
```

Once served, you may query the model with the following command.

Sample request

```bash
curl -X POST \
  http://localhost:8501/v1/models/glove:predict \
  -d '{"instances": [{"row_token": "man", "column_token": "man"}]}'
```

Sample response

```
{
    "predictions": [
        {
            
            "top_k_row_string": [
                "man",
                "woman",
                "person",
                "men",
                "women",
                "children",
                "girl",
                "son",
                "father",
                "god",
                ...
            ],
            "top_k_row_similarity": [
                1,
                0.730189,
                0.644536,
                0.614566,
                0.558976,
                0.553486,
                0.550555,
                0.533169,
                0.521285,
                0.518982,
                ...
            ],
            "row_embed": [
                -0.0735308,
                0.14473,
                0.587398,
                0.354783,
                0.148979,
                0.302668,
                -0.410081,
                -0.158809,
                0.0883906,
                0.0459743,
                ...
            ],
            "row_bias": 0.189876,
            "top_k_column_string": [
                "man",
                "woman",
                "person",
                "child",
                "men",
                "love",
                "girl",
                "son",
                "shot",
                "god",
                ...
            ],
            "top_k_column_similarity": [
                1,
                0.691862,
                0.648038,
                0.590263,
                0.554949,
                0.550537,
                0.536444,
                0.533546,
                0.528475,
                0.522893,
                ...
            ],
            "column_embed": [
                0.24108,
                0.184525,
                0.557452,
                0.501938,
                -0.263059,
                0.214702,
                -0.60232,
                0.299381,
                -0.241118,
                0.0867298,
                ...
            ],
            "column_bias": 0.218183,
            "embed_norm_product": 0.362292
        }
    ]
}
```

## Distributed

For distributed model training and evaluation, please refer to [distributed](distributed.md).

## References

- Mahoney, M. (2006). Large Text Compression Benchmark. Retrieved from [http://mattmahoney.net/dc/text.html](http://mattmahoney.net/dc/text.html).
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. \[[pdf](https://nlp.stanford.edu/pubs/glove.pdf)\]\[[bib](https://nlp.stanford.edu/pubs/glove.bib)\]

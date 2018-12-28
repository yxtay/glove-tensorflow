# GloVe Model in TensorFlow
Implementation of GloVe using TensorFlow estimator API.

## Setup

```bash
# clone repo
git clone git@github.com:yxtay/glove-tensorflow.git && cd recommender-tensorflow

# create conda environment
conda env create -f=environment.yml

# activate environment
source activate dl
```

## Download & Process Data

The [text8 dataset](http://mattmahoney.net/dc/textdata.html) is used for demonstration purposes.
The following script downloads the data, processes it to prepare the vocabulary and cooccurrence matrix. The data is serialised to `csv`.

```bash
python -m src.data.text8
```

**Usage**

```bash
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

**Usage**

```bash
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

## TensorFlow Serving

The trained and serialised model may be served with TensorFlow Serving.

```bash
docker run --rm -p 8500:8500 -p 8501:8501 \
--mount type=bind,source=$(pwd)/checkpoints/glove/export/exporter,target=/models/glove \
-e MODEL_NAME=glove -t tensorflow/serving
```

**Model signature**

```bash
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
    outputs['top_k_column_string'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 100)
        name: similarity/column_token_string_lookup_Lookup:0
    outputs['top_k_column_similarity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: similarity/top_k_sim_column_token:0
    outputs['top_k_row_string'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 100)
        name: similarity/row_token_string_lookup_Lookup:0
    outputs['top_k_row_similarity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: similarity/top_k_sim_row_token:0
  Method name is: tensorflow/serving/predict
```

Once served, you may query the model with the following command.

Sample request

```bash
curl -X POST \
  http://localhost:8501/v1/models/glove:predict \
  -d '{"instances": [{"row_name": "man", "column_name": "man"}]}'
```

Sample response

```bash
{
    "predictions": [
        {
            "top_k_row_string": [
                "man",
                "woman",
                "young",
                "leaving",
                "named",
                "love",
                "child",
                "wrote",
                "children",
                "death",
                ...
            ],
            "top_k_row_similarity": [
                1,
                0.858384,
                0.843938,
                0.827622,
                0.823886,
                0.817748,
                0.812745,
                0.811423,
                0.805291,
                0.804582,
                ...
            ],
            "row_embed": [
                0.110241,
                -0.0781973,
                0.252237,
                0.173164,
                -0.0698477,
                0.121707,
                0.199838,
                0.282855,
                -0.336472,
                0.11848,
                ...
            ],
            "row_bias": 0.00140062,
            "top_k_column_string": [
                "man",
                "woman",
                "wrote",
                "named",
                "child",
                "young",
                "children",
                "person",
                "love",
                "god",
                ...
            ],
            "top_k_column_similarity": [
                1,
                0.870042,
                0.822171,
                0.818766,
                0.810546,
                0.802626,
                0.799576,
                0.795452,
                0.787946,
                0.787853,
                ...
            ],
            "column_embed": [
                0.131286,
                -0.220142,
                0.164308,
                0.172627,
                -0.107009,
                0.154377,
                -0.149073,
                0.365269,
                -0.288956,
                -0.035981,
                ...
            ],
            "column_bias": -0.0476088,
            "embed_norm_product": 0.73634,
        }
    ]
}
```

## References

- Mahoney, M. (2006). Large Text Compression Benchmark. Retrieved from [http://mattmahoney.net/dc/text.html](http://mattmahoney.net/dc/text.html).
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. \[[pdf](https://nlp.stanford.edu/pubs/glove.pdf)\]\[[bib](https://nlp.stanford.edu/pubs/glove.bib)\]

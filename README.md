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
    inputs['column_name'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: Placeholder_1:0
    inputs['row_name'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: Placeholder:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['column_bias'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: mf/column_bias_lookup/Identity:0
    outputs['column_embed'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 64)
        name: mf/column_embed_lookup/Identity:0
    outputs['embed_product'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: mf/Sum:0
    outputs['row_bias'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: mf/row_bias_lookup/Identity:0
    outputs['row_embed'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 64)
        name: mf/row_embed_lookup/Identity:0
    outputs['top_k_column_names'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 100)
        name: similarity/column_string_lookup_Lookup:0
    outputs['top_k_column_similarity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: similarity/TopKV2_1:0
    outputs['top_k_row_names'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 100)
        name: similarity/row_string_lookup_Lookup:0
    outputs['top_k_row_similarity'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 100)
        name: similarity/TopKV2:0
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
            "top_k_row_names": [
                "man",
                "woman",
                "person",
                "girl",
                "young",
                "doctor",
                "god",
                "child",
                "men",
                "love",
                ...
            ],
            "top_k_row_similarity": [
                1,
                0.747999,
                0.684991,
                0.634778,
                0.587861,
                0.563282,
                0.557204,
                0.553089,
                0.552128,
                0.548473,
                0.529555,
                ...
            ],
            "row_embed": [
                -0.540615,
                0.105328,
                -0.0805282,
                -0.256874,
                -0.0443556,
                -0.274111,
                -0.173894,
                0.277241,
                -0.265513,
                -0.159817,
                ...
            ],
            "row_bias": 0.0635015,
            "top_k_column_names": [
                "man",
                "woman",
                "men",
                "person",
                "girl",
                "children",
                "young",
                "god",
                "child",
                "son",
                ...
            ],
            "top_k_column_similarity": [
                1,
                0.74111,
                0.668318,
                0.654749,
                0.654203,
                0.59045,
                0.589824,
                0.581404,
                0.581264,
                0.546073,
                ...
            ],
            "column_embed": [
                -0.24833,
                0.162511,
                -0.329551,
                -0.438676,
                0.193193,
                -0.210588,
                -0.237587,
                0.385656,
                -0.187433,
                0.224222,
                ...
            ],
            "column_bias": -0.0147059,
            "embed_product": 2.05853
        }
    ]
}
```

## References

- Mahoney, M. (2006). Large Text Compression Benchmark. Retrieved from [http://mattmahoney.net/dc/text.html](http://mattmahoney.net/dc/text.html).
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. \[[pdf](https://nlp.stanford.edu/pubs/glove.pdf)\]\[[bib](https://nlp.stanford.edu/pubs/glove.bib)\]

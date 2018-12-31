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

**Sample data**

| row_token_id | column_token_id | interaction | row_token  | column_token | glove_weight | glove_value |
|--------------|-----------------|-------------|------------|--------------|--------------|-------------|
| 608          | 247             | 1.5833      | town       | free         | 0.0331       | 0.4595      |
| 1070         | 53              | 3.4166      | magazine   | can          | 0.0591       | 1.2286      |
| 7050         | 5239            | 2.0         | syllables  | numbered     | 0.0395       | 0.6931      |
| 5315         | 1391            | 1.4166      | malta      | geography    | 0.0305       | 0.3483      |
| 364          | 0               | 1481.7500   | never      | \<UNK\>        | 1.0          | 7.3009      |
| 269          | 144             | 15.6499     | line       | form         | 0.1850       | 2.7504      |
| 631          | 2895            | 1.5333      | characters | gary         | 0.0324       | 0.4274      |
| 1437         | 74              | 7.1166      | attacks    | would        | 0.1024       | 1.9624      |
| 6788         | 4522            | 2.0         | sexually   | adults       | 0.0395       | 0.6931      |

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

## TensorFlow Serving

The trained and serialised model may be served with TensorFlow Serving.

```bash
CHECKPOINT_PATH=checkpoints/glove/export/exporter

docker run --rm -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=$(pwd)/${CHECKPOINT_PATH},target=/models/glove \
  -e MODEL_NAME=glove -t tensorflow/serving
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

## References

- Mahoney, M. (2006). Large Text Compression Benchmark. Retrieved from [http://mattmahoney.net/dc/text.html](http://mattmahoney.net/dc/text.html).
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. \[[pdf](https://nlp.stanford.edu/pubs/glove.pdf)\]\[[bib](https://nlp.stanford.edu/pubs/glove.bib)\]

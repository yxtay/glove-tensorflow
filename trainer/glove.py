import json
import os
from argparse import ArgumentParser

import tensorflow as tf

from trainer.config import COL_ID, CONFIG, ROW_ID, VOCAB_TXT
from trainer.utils import (
    get_eval_spec, get_exporter, get_keras_dataset_input_fn, get_minimise_op, get_optimizer, get_run_config,
    get_serving_input_fn, get_train_spec,
)


def get_field_variables(features, field_values, vocab_txt=VOCAB_TXT, embedding_size=64):
    name = field_values["name"]
    with tf.name_scope(name):
        # create field variables
        with tf.io.gfile.GFile(vocab_txt) as f:
            vocab = f.readlines()
        vocab_size = len(vocab)

        string_id_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            vocab_txt,
            tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
        ), 0, name=name + "_id_lookup")
        initializer = tf.random_uniform_initializer()

        # variables
        field_embeddings = tf.Variable(initializer([vocab_size, embedding_size]), name=name + "_embeddings")
        field_biases = tf.Variable(initializer([vocab_size]), name=name + "_biases")

        # get field values
        field_idx = string_id_table.lookup(features[name])
        # [None]
        field_embed = tf.nn.embedding_lookup(
            field_embeddings,
            field_idx,
            name=name + "_embed_lookup"
        )
        # [None, embedding_size]
        field_bias = tf.nn.embedding_lookup(
            field_biases,
            field_idx,
            name=name + "_bias_lookup"
        )
        # [None, 1]
    field_values.update({
        "id": features[name],
        "embeddings": field_embeddings,
        "biases": field_biases,
        "embed": field_embed,
        "bias": field_bias,
    })
    return field_values


def get_similarity(field_values, vocab_txt=VOCAB_TXT, k=100):
    name = field_values["name"]
    with tf.name_scope(field_values["name"]):
        id_string_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            vocab_txt,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
            tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        ), "<UNK>", name=name + "_string_lookup")

        field_embed_norm = tf.math.l2_normalize(field_values["embed"], 1)
        # [None, embedding_size]
        field_embeddings_norm = tf.math.l2_normalize(field_values["embeddings"], 1)
        # [vocab_size, embedding_size]
        field_cosine_sim = tf.matmul(field_embed_norm, field_embeddings_norm, transpose_b=True)
        # [None, mapping_size]
        field_top_k_sim, field_top_k_idx = tf.math.top_k(
            field_cosine_sim,
            k=k,
            name="top_k_sim_" + field_values["name"]
        )
        # [None, k], [None, k]
        field_top_k_string = id_string_table.lookup(tf.cast(field_top_k_idx, tf.int64))
        # [None, k]
    field_values.update({
        "embed_norm": field_embed_norm,
        "embeddings_norm": field_embeddings_norm,
        "top_k_sim": field_top_k_sim,
        "top_k_idx": field_top_k_idx,
        "top_k_string": field_top_k_string
    })
    return field_values


def model_fn(features, labels, mode, params):
    field_names = params["field_names"]
    mappings = params["mappings"]
    embedding_size = params.get("embedding_size", 64)
    optimizer_name = params.get("optimizer", "Adam")
    learning_rate = params.get("learning_rate", 0.001)
    k = params.get("k", 100)
    l2_reg = params.get("l2_reg", 0.)

    row_id_variables = {"name": field_names["row_id"], "mapping": mappings[field_names["row_id"]]}
    col_id_variables = {"name": field_names["col_id"], "mapping": mappings[field_names["col_id"]]}

    with tf.name_scope("mf"):
        # global bias
        l2_regularizer = tf.keras.regularizers.l2(1.0)
        global_bias = tf.get_variable("global_bias", [], regularizer=l2_regularizer)
        # []
        tf.summary.scalar("global_bias", global_bias)
        # row mapping, embeddings and biases
        row_id_variables = get_field_variables(features, row_id_variables, embedding_size)
        # column mapping, embeddings and biases
        col_id_variables = get_field_variables(features, col_id_variables, embedding_size)

        # matrix factorisation
        embed_product = tf.reduce_sum(tf.multiply(row_id_variables["embed"], col_id_variables["embed"]), 1)
        # [None, 1]
        predict_value = tf.add_n([
            tf.ones_like(embed_product) * global_bias,
            row_id_variables["bias"],
            col_id_variables["bias"],
            embed_product
        ])
        # [None, 1]

    # prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        # calculate similarity
        with tf.name_scope("similarity"):
            # row similarity
            row_id_variables = get_similarity(row_id_variables, k)
            # col similarity
            col_id_variables = get_similarity(col_id_variables, k)

            embed_norm_product = tf.reduce_sum(tf.multiply(
                row_id_variables["embed_norm"],
                col_id_variables["embed_norm"]
            ), 1)

        predictions = {
            "predicted_value": predict_value,
            "row_id": row_id_variables["id"],
            "row_embed": row_id_variables["embed"],
            "row_bias": row_id_variables["bias"],
            "col_id": col_id_variables["id"],
            "col_embed": col_id_variables["embed"],
            "col_bias": col_id_variables["bias"],
            "embed_norm_product": embed_norm_product,
            "top_k_row_similarity": row_id_variables["top_k_sim"],
            "top_k_row_string": row_id_variables["top_k_string"],
            "top_k_col_similarity": col_id_variables["top_k_sim"],
            "top_k_col_string": col_id_variables["top_k_string"],
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # evaluation
    with tf.name_scope("regularised_loss"):
        mse_loss = tf.keras.losses.MeanSquaredError()(
            labels[field_names["value"]], predict_value, labels[field_names["weight"]]
        )
        # []
        loss = mse_loss + l2_reg * tf.losses.get_regularization_loss()
        # []
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    # training
    optimizer = get_optimizer(optimizer_name, learning_rate)
    train_op = get_minimise_op(loss, optimizer)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def get_estimator(job_dir, params):
    run_config = get_run_config()
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=job_dir,
        config=run_config,
        params=params
    )
    return estimator


def get_predict_input_fn(mapping):
    row_id = CONFIG["field_names"]["row_id"]
    col_id = CONFIG["field_names"]["col_id"]
    output_types = {row_id: tf.string, col_id: tf.string}

    def input_generator():
        for el in mapping:
            yield {row_id: el, col_id: el}

    def input_fn():
        dataset = tf.data.Dataset.from_generator(input_generator, output_types)
        dataset = dataset.batch(1)
        return dataset

    return input_fn


def train_and_evaluate(args):
    # paths
    train_csv = args.train_csv
    vocab_json = args.vocab_json
    job_dir = args.job_dir
    restore = args.restore
    # model
    embedding_size = args.embedding_size
    l2_reg = args.l2_reg
    k = args.k
    # training
    batch_size = args.batch_size
    train_steps = args.train_steps

    # init
    tf.logging.set_verbosity(tf.logging.INFO)
    if not restore and tf.io.gfile.exists(job_dir):
        tf.io.gfile.rmtree(job_dir)
    tf.io.gfile.makedirs(job_dir)

    # load vocab
    with tf.io.gfile.GFile(vocab_json) as f:
        vocab = json.load(f)

    # save params
    params = {
        "field_names": CONFIG["field_names"],
        "mappings": {
            ROW_ID: vocab,
            COL_ID: vocab,
        },
        "embedding_size": embedding_size,
        "l2_reg": l2_reg,
        "k": k,
    }
    params_json = os.path.join(job_dir, "params.json")
    with tf.io.gfile.GFile(params_json, "w") as f:
        json.dump(params, f, indent=2)

    # estimator
    estimator = get_estimator(job_dir, params)

    # train spec
    input_fn_args = CONFIG["input_fn_args"]
    train_input_fn = get_keras_dataset_input_fn(file_pattern=train_csv, batch_size=batch_size, **input_fn_args)
    train_spec = get_train_spec(train_input_fn, train_steps)

    # eval spec
    eval_input_fn = get_keras_dataset_input_fn(file_pattern=train_csv, mode=tf.estimator.ModeKeys.EVAL,
                                               batch_size=batch_size, **input_fn_args)
    serving_input_fn_args = CONFIG["serving_input_fn_args"]
    exporter = get_exporter(get_serving_input_fn(**serving_input_fn_args))
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default="data/interaction.csv",
        help="path to the training csv data (default: %(default)s)"
    )
    parser.add_argument(
        "--vocab-json",
        default="data/vocab.json",
        help="path to the vocab json (default: %(default)s)"
    )
    parser.add_argument(
        "--job-dir",
        default="checkpoints/glove",
        help="job directory (default: %(default)s)"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="whether to restore from JOB_DIR"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=64,
        help="embedding size (default: %(default)s)"
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=0.01,
        help="scale of l2 regularisation (default: %(default)s)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="k for top k similarity (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=16384,
        help="number of training steps (default: %(default)s)"
    )
    args = parser.parse_args()

    train_and_evaluate(args)

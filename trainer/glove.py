import json
import shutil
from argparse import ArgumentParser

import tensorflow as tf

from trainer.conf_utils import get_run_config, get_train_spec, get_exporter, get_eval_spec
from trainer.model_utils import get_optimizer
from trainer.text8 import get_input_fn, serving_input_fn


def model_fn(features, labels, mode, params):
    field_names = params.get("field_names",
                             {
                                 "row": "row_name",
                                 "column": "column_name",
                                 "weight": "interaction_weight",
                             })
    mapping = params["mapping"]
    embedding_size = params.get("embedding_size", 4)
    optimizer_name = params.get("optimizer", "Adam")
    learning_rate = params.get("learning_rate", 0.001)
    k = params.get("k", 100)

    with tf.name_scope("mf"):
        # row mapping and embeddings
        row_mapping = mapping["row"]
        row_string_id_table = (tf.contrib.lookup
                             .index_table_from_tensor(row_mapping, default_value=0, name="row_id_lookup"))
        row_dim = len(row_mapping)
        row_embeddings = tf.get_variable("row_embeddings", [row_dim, embedding_size])
        row_biases = tf.get_variable("row_biases", [row_dim])
        tf.summary.histogram("row_bias", row_biases)
        # column mapping and embeddings
        column_mapping = mapping["column"]
        column_string_id_table = (tf.contrib.lookup
                                .index_table_from_tensor(column_mapping, default_value=0, name="column_id_lookup"))
        column_dim = len(column_mapping)
        column_embeddings = tf.get_variable("column_embeddings", [column_dim, embedding_size])
        column_biases = tf.get_variable("column_biases", [column_dim])
        tf.summary.histogram("column_bias", column_biases)
        # global bias
        global_bias = tf.get_variable("global_bias", [])
        # []
        tf.summary.scalar("global_bias", global_bias)

        # row values
        row_id = row_string_id_table.lookup(features[field_names["row"]])
        # [None]
        row_embed = tf.nn.embedding_lookup(row_embeddings, row_id, name="row_embed_lookup")
        # [None, embedding_size]
        row_bias = tf.nn.embedding_lookup(row_biases, row_id, name="row_bias_lookup")
        # [None, 1]
        # column values
        column_id = column_string_id_table.lookup(features[field_names["column"]])
        # [None]
        column_embed = tf.nn.embedding_lookup(column_embeddings, column_id, name="column_embed_lookup")
        # [None, embedding_size]
        column_bias = tf.nn.embedding_lookup(column_biases, column_id, name="column_bias_lookup")
        # [None, 1]

        # matrix factorisation
        embed_product = tf.reduce_sum(tf.multiply(row_embed, column_embed), 1)
        # [None, 1]
        predicted_value = tf.add(global_bias, tf.add_n([row_bias, column_bias, embed_product]))
        # [None, 1]

    # prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        # calculate similarity
        with tf.name_scope("similarity"):
            # row similarity
            row_embed_norm = tf.math.l2_normalize(row_embed, 1)
            # [None, embedding_size]
            row_embeddings_norm = tf.math.l2_normalize(row_embeddings, 1)
            # [vocab_size, embedding_size]
            row_cosine_sim = tf.matmul(row_embed_norm, row_embeddings_norm, transpose_b=True)
            # [None, vocab_size]

            top_k_row_sim, top_k_row_idx = tf.math.top_k(row_cosine_sim, k=k)
            # [None, k], [None, k]f
            row_id_string_table = (tf.contrib.lookup
                                   .index_to_string_table_from_tensor(row_mapping, name="row_string_lookup"))
            top_k_row_names = row_id_string_table.lookup(tf.cast(top_k_row_idx, tf.int64))
            # [None, k]

            # column similarity
            column_embed_norm = tf.math.l2_normalize(column_embed, 1)
            # [None, embedding_size]
            column_embeddings_norm = tf.math.l2_normalize(column_embeddings, 1)
            # [vocab_size, embedding_size]
            column_cosine_sim = tf.matmul(column_embed_norm, column_embeddings_norm, transpose_b=True)
            # [None, vocab_size]

            top_k_column_sim, top_k_column_idx = tf.math.top_k(column_cosine_sim, k=k)
            # [None, k], [None, k]
            column_id_string_table = (tf.contrib.lookup
                                      .index_to_string_table_from_tensor(column_mapping, name="column_string_lookup"))
            top_k_column_names = column_id_string_table.lookup(tf.cast(top_k_column_idx, tf.int64))
            # [None, k]

        predictions = {
            "row_embed": row_embed,
            "row_bias": row_bias,
            "column_embed": column_embed,
            "column_bias": column_bias,
            "embed_product": embed_product,
            "top_k_row_similarity": top_k_row_sim,
            "top_k_row_names": top_k_row_names,
            "top_k_column_similarity": top_k_column_sim,
            "top_k_column_names": top_k_column_names,
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # evaluation
    with tf.name_scope("mse"):
        loss = tf.losses.mean_squared_error(labels,
                                            predicted_value,
                                            features[field_names["weight"]])
        # []
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    # training
    with tf.name_scope("train"):
        optimizer = get_optimizer(optimizer_name, learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def train_and_evaluate(args):
    # paths
    train_csv = args.train_csv
    vocab_json = args.vocab_json
    job_dir = args.job_dir
    restore = args.restore
    # model
    embedding_size = args.embedding_size
    k = args.k
    # training
    batch_size = args.batch_size
    train_steps = args.train_steps

    # init
    tf.logging.set_verbosity(tf.logging.INFO)
    if not restore:
        shutil.rmtree(job_dir, ignore_errors=True)

    # load vocab
    with open(vocab_json) as f:
        vocab = json.load(f)

    # estimator
    run_config = get_run_config()
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=job_dir,
        config=run_config,
        params={
            "mapping": {"row": vocab, "column": vocab},
            "embedding_size": embedding_size,
            "k": k,
        }
    )

    # train spec
    train_input_fn = get_input_fn(train_csv, batch_size=batch_size)
    train_spec = get_train_spec(train_input_fn, train_steps)

    # eval spec
    eval_input_fn = get_input_fn(train_csv, tf.estimator.ModeKeys.EVAL, batch_size=batch_size)
    exporter = get_exporter(serving_input_fn)
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train-csv", default="data/interaction.csv",
                        help="path to the training csv data (default: %(default)s)")
    parser.add_argument("--vocab-json", default="data/vocab.json",
                        help="path to the vocab json (default: %(default)s)")
    parser.add_argument("--job-dir", default="checkpoints/glove",
                        help="job directory (default: %(default)s)")
    parser.add_argument("--restore", action="store_true",
                        help="whether to restore from JOB_DIR")
    parser.add_argument("--embedding-size", type=int, default=64,
                        help="embedding size (default: %(default)s)")
    parser.add_argument("--k", type=int, default=100,
                        help="k for top k similarity (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size (default: %(default)s)")
    parser.add_argument("--train-steps", type=int, default=20000,
                        help="number of training steps (default: %(default)s)")
    args = parser.parse_args()

    train_and_evaluate(args)

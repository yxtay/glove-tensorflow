import numpy as np
import tensorflow as tf

from trainer.config import (
    COL_ID, CONFIG, EMBEDDING_SIZE, L2_REG, LEARNING_RATE, OPTIMIZER, ROW_ID, TOP_K, VOCAB_TXT, WEIGHT,
)
from trainer.glove_utils import get_id_string_table, get_string_id_table, init_params, parse_args
from trainer.utils import (
    file_lines, get_csv_input_fn, get_eval_spec, get_exporter, get_optimizer, get_run_config, get_serving_input_fn,
    get_train_spec,
)

v1 = tf.compat.v1


def get_regularized_variable(name, shape=(), l2_reg=1.0, **kwargs):
    l2_reg = l2_reg / np.prod(shape)
    regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=l2_reg)
    variables = v1.get_variable(name, shape, regularizer=regularizer, **kwargs)
    return variables


def get_field_values(features, field_values, vocab_txt=VOCAB_TXT, embedding_size=64, l2_reg=1.0):
    name = field_values["name"]
    with tf.name_scope(name):
        # create field variables
        vocab_size = file_lines(vocab_txt)
        string_id_table = get_string_id_table(vocab_txt)

        # variables
        field_embeddings = get_regularized_variable(name + "_embeddings", [vocab_size, embedding_size], l2_reg)
        field_biases = get_regularized_variable(name + "_biases", [vocab_size], l2_reg)
        v1.summary.histogram("biases", field_biases)

        # get field values
        field_idx = string_id_table.lookup(features[name])
        # [None]
        field_embed = tf.nn.embedding_lookup(field_embeddings, field_idx, name=name + "_embed_lookup")
        # [None, embedding_size]
        field_bias = tf.nn.embedding_lookup(field_biases, field_idx, name=name + "_bias_lookup")
        # [None, 1]
    field_values.update({
        "id": tf.identity(features[name]),
        "embeddings": field_embeddings,
        "biases": field_biases,
        "embed": field_embed,
        "bias": field_bias,
    })
    return field_values


def get_similarity(field_values, vocab_txt=VOCAB_TXT, top_k=TOP_K):
    name = field_values["name"]
    with tf.name_scope(name):
        id_string_table = get_id_string_table(vocab_txt)

        field_embed_norm = tf.math.l2_normalize(field_values["embed"], -1)
        # [None, embedding_size]
        field_embeddings_norm = tf.math.l2_normalize(field_values["embeddings"], -1)
        # [vocab_size, embedding_size]
        field_cosine_sim = tf.matmul(field_embed_norm, field_embeddings_norm, transpose_b=True)
        # [None, mapping_size]
        field_top_k = tf.math.top_k(field_cosine_sim, k=top_k, name="top_k_sim_" + name)
        field_top_k_sim, field_top_k_idx = field_top_k
        # [None, k], [None, k]
        field_top_k_string = id_string_table.lookup(tf.cast(field_top_k_idx, tf.int64), name=name + "_string_lookup")
        # [None, k]
    field_values.update({
        "embed_norm": field_embed_norm,
        "embeddings_norm": field_embeddings_norm,
        "top_k_sim": field_top_k_sim,
        "top_k_string": field_top_k_string
    })
    return field_values


def model_fn(features, labels, mode, params):
    vocab_txt = params.get("vocab_txt", VOCAB_TXT)
    embedding_size = params.get("embedding_size", EMBEDDING_SIZE)
    l2_reg = params.get("l2_reg", L2_REG)
    optimizer_name = params.get("optimizer", OPTIMIZER)
    learning_rate = params.get("learning_rate", LEARNING_RATE)
    top_k = params.get("top_k", TOP_K)

    row_values = {"name": ROW_ID}
    col_values = {"name": COL_ID}

    with tf.name_scope("mf"):
        # global bias
        global_bias = get_regularized_variable("global_bias", initializer=tf.zeros_initializer, l2_reg=l2_reg)
        # []
        v1.summary.scalar("global_bias", global_bias)
        # row mapping, embeddings and biases
        row_values = get_field_values(features, row_values, vocab_txt, embedding_size, l2_reg)
        # column mapping, embeddings and biases
        col_values = get_field_values(features, col_values, vocab_txt, embedding_size, l2_reg)

        # matrix factorisation
        embed_product = tf.reduce_sum(tf.multiply(row_values["embed"], col_values["embed"]), 1)
        # [None, 1]
        logits = tf.add_n([
            tf.ones_like(embed_product) * global_bias,
            row_values["bias"],
            col_values["bias"],
            embed_product
        ])
        logits = tf.expand_dims(logits, -1)
        # [None, 1]

    # prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        # calculate similarity
        with tf.name_scope("similarity"):
            # row similarity
            row_values = get_similarity(row_values, vocab_txt, top_k)
            # col similarity
            col_values = get_similarity(col_values, vocab_txt, top_k)

        predictions = {
            "row_id": row_values["id"],
            "row_embed": row_values["embed"],
            "row_bias": row_values["bias"],
            "col_id": col_values["id"],
            "col_embed": col_values["embed"],
            "col_bias": col_values["bias"],
            "top_k_row_similarity": row_values["top_k_sim"],
            "top_k_row_string": row_values["top_k_string"],
            "top_k_col_similarity": col_values["top_k_sim"],
            "top_k_col_string": col_values["top_k_string"],
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # training
    optimizer = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    # head
    head = tf.estimator.RegressionHead(weight_column=WEIGHT)
    return head.create_estimator_spec(
        features, mode, logits,
        labels=labels,
        optimizer=optimizer,
        trainable_variables=v1.trainable_variables(),
        update_ops=v1.get_collection(v1.GraphKeys.UPDATE_OPS),
        regularization_losses=v1.losses.get_regularization_losses(),
    )


def get_estimator(params):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params["job_dir"],
        config=get_run_config(),
        params=params
    )
    return estimator


def get_predict_input_fn(vocab_txt):
    output_types = {ROW_ID: tf.string, COL_ID: tf.string}

    def input_generator():
        with tf.io.gfile.GFile(vocab_txt) as f:
            for line in f:
                line = line.strip()
                yield {ROW_ID: line, COL_ID: line}

    def input_fn():
        dataset = tf.data.Dataset.from_generator(input_generator, output_types)
        dataset = dataset.batch(1)
        return dataset

    return input_fn


def main():
    args = parse_args()
    params = init_params(args.__dict__)

    # estimator
    estimator = get_estimator(params)

    # train spec
    dataset_args = {
        "file_pattern": params["train_csv"],
        "batch_size": params["batch_size"],
        **CONFIG["dataset_args"],
    }
    train_input_fn = get_csv_input_fn(**dataset_args, num_epochs=None)
    eval_input_fn = get_csv_input_fn(**dataset_args)

    # eval spec
    train_spec = get_train_spec(train_input_fn, params["train_steps"])
    exporter = get_exporter(get_serving_input_fn(**CONFIG["serving_input_fn_args"]))
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

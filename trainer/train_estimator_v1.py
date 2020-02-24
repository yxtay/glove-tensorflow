import numpy as np
import tensorflow as tf

from trainer.config import (
    COL_NAME, EMBEDDING_SIZE, L2_REG, LEARNING_RATE, OPTIMIZER, ROW_NAME, TARGET_NAME, TOP_K, VOCAB_TXT, WEIGHT_NAME,
    parse_args,
)
from trainer.data_utils import get_csv_input_fn, get_serving_input_fn
from trainer.model_utils import get_id_string_table, get_string_id_table
from trainer.train_utils import get_estimator, get_eval_spec, get_exporter, get_optimizer, get_train_spec
from trainer.utils import cosine_similarity, file_lines


def get_regularized_variable(name, shape=(), l2_reg=1.0, **kwargs):
    l2_reg = l2_reg / np.prod(shape)
    regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=l2_reg)
    variables = tf.compat.v1.get_variable(name, shape, regularizer=regularizer, **kwargs)
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
        tf.compat.v1.summary.histogram("biases", field_biases)

        # get field values
        field_idx = string_id_table.lookup(features[name])
        # [None]
        field_embed = tf.nn.embedding_lookup(field_embeddings, field_idx, name=name + "_embed_lookup")
        # [None, embedding_size]
        field_bias = tf.nn.embedding_lookup(field_biases, field_idx, name=name + "_bias_lookup")
        # [None, 1]
    field_values.update({
        "string": tf.identity(features[name]),
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

        field_cosine_sim = cosine_similarity(field_values["embed"], field_values["embeddings"])
        # [None, mapping_size]
        field_top_k = tf.math.top_k(field_cosine_sim, k=top_k, name="top_k_sim_" + name)
        field_top_k_sim, field_top_k_idx = field_top_k
        # [None, k], [None, k]
        field_top_k_string = id_string_table.lookup(tf.cast(field_top_k_idx, tf.int64), name=name + "_string_lookup")
        # [None, k]
    field_values.update({
        "top_k_sim": field_top_k_sim,
        "top_k_string": field_top_k_string
    })
    return field_values


def model_fn(features, labels, mode, params):
    row_name = params.get("row_name", ROW_NAME)
    col_name = params.get("col_name", COL_NAME)
    target_name = params.get("target_name", TARGET_NAME)
    weight_name = params.get("weight_name", WEIGHT_NAME)
    vocab_txt = params.get("vocab_txt", VOCAB_TXT)
    embedding_size = params.get("embedding_size", EMBEDDING_SIZE)
    l2_reg = params.get("l2_reg", L2_REG)
    optimizer_name = params.get("optimizer", OPTIMIZER)
    learning_rate = params.get("learning_rate", LEARNING_RATE)
    top_k = params.get("top_k", TOP_K)

    row_values = {"name": row_name}
    col_values = {"name": col_name}

    with tf.name_scope("mf"):
        # global bias
        global_bias = get_regularized_variable("global_bias", initializer=tf.zeros_initializer, l2_reg=l2_reg)
        # []
        tf.compat.v1.summary.scalar("global_bias", global_bias)
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
            "row_string": row_values["string"],
            "row_embed": row_values["embed"],
            "col_string": col_values["string"],
            "col_embed": col_values["embed"],
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
    head = tf.estimator.RegressionHead(weight_column=weight_name)
    return head.create_estimator_spec(
        features, mode, logits,
        labels=labels[target_name],
        optimizer=optimizer,
        trainable_variables=tf.compat.v1.trainable_variables(),
        update_ops=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS),
        regularization_losses=tf.compat.v1.losses.get_regularization_losses(),
    )


def get_predict_input_fn(vocab_txt):
    output_types = {ROW_NAME: tf.string, COL_NAME: tf.string}

    def input_generator():
        with tf.io.gfile.GFile(vocab_txt) as f:
            for line in f:
                line = line.strip()
                yield {ROW_NAME: line, COL_NAME: line}

    def input_fn():
        dataset = tf.data.Dataset.from_generator(input_generator, output_types)
        dataset = dataset.batch(1)
        return dataset

    return input_fn


def main():
    params = parse_args()

    # estimator
    estimator = get_estimator(model_fn, params)

    # input functions
    train_input_fn = get_csv_input_fn(**params["input_fn_args"], num_epochs=None)
    eval_input_fn = get_csv_input_fn(**params["input_fn_args"])

    # train, eval spec
    train_spec = get_train_spec(train_input_fn, params["train_steps"])
    exporter = get_exporter(get_serving_input_fn(**params["serving_input_fn_args"]))
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

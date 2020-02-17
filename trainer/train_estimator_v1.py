import tensorflow as tf

from trainer.config import COL_ID, CONFIG, EMBEDDING_SIZE, L2_REG, LEARNING_RATE, ROW_ID, TARGET, VOCAB_TXT
from trainer.glove_utils import get_id_string_table, get_string_id_table, init_params, parse_args
from trainer.utils import (
    file_lines, get_eval_spec, get_exporter, get_keras_dataset_input_fn, get_run_config, get_serving_input_fn,
    get_train_spec,
)

v1 = tf.compat.v1


def get_field_values(features, field_values, vocab_txt=VOCAB_TXT, embedding_size=64):
    name = field_values["name"]
    with tf.name_scope(name):
        # create field variables
        vocab_size = file_lines(vocab_txt)
        string_id_table = get_string_id_table(vocab_txt)

        # variables
        l2_regularizer = tf.keras.regularizers.l2(1.0 / (vocab_size * embedding_size))
        field_embeddings = v1.get_variable(
            name + "_embeddings",
            [vocab_size, embedding_size],
            regularizer=l2_regularizer
        )
        l2_regularizer = tf.keras.regularizers.l2(1.0 / (vocab_size * embedding_size))
        field_biases = v1.get_variable(name + "_biases", [vocab_size], regularizer=l2_regularizer)
        v1.summary.histogram(field_values["name"] + "_bias", field_biases)

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


def get_similarity(field_values, vocab_txt=VOCAB_TXT, k=100):
    name = field_values["name"]
    with tf.name_scope(name):
        id_string_table = get_id_string_table(vocab_txt)

        field_embed_norm = tf.math.l2_normalize(field_values["embed"], 1)
        # [None, embedding_size]
        field_embeddings_norm = tf.math.l2_normalize(field_values["embeddings"], 1)
        # [vocab_size, embedding_size]
        field_cosine_sim = tf.matmul(field_embed_norm, field_embeddings_norm, transpose_b=True)
        # [None, mapping_size]
        field_top_k = tf.math.top_k(field_cosine_sim, k=k, name="top_k_sim_" + name)
        field_top_k_sim, field_top_k_idx = field_top_k
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
    vocab_txt = params.get("vocab_txt", VOCAB_TXT)
    embedding_size = params.get("embedding_size", EMBEDDING_SIZE)
    l2_reg = params.get("l2_reg", L2_REG)
    learning_rate = params.get("learning_rate", LEARNING_RATE)
    k = params.get("k", 100)

    if set(features.keys()) == {"features", "sample_weights"}:
        sample_weights = features["sample_weights"]
        features = features["features"]
    else:
        sample_weights = {TARGET: None}

    row_values = {"name": ROW_ID}
    col_values = {"name": COL_ID}

    with tf.name_scope("mf"):
        # global bias
        l2_regularizer = tf.keras.regularizers.l2(1.0)
        global_bias = v1.get_variable("global_bias", [], regularizer=l2_regularizer)
        # []
        v1.summary.scalar("global_bias", global_bias)
        # row mapping, embeddings and biases
        row_values = get_field_values(features, row_values, vocab_txt, embedding_size)
        # column mapping, embeddings and biases
        col_values = get_field_values(features, col_values, vocab_txt, embedding_size)

        # matrix factorisation
        embed_product = tf.reduce_sum(tf.multiply(row_values["embed"], col_values["embed"]), 1)
        # [None, 1]
        predict_value = tf.add_n([
            tf.ones_like(embed_product) * global_bias,
            row_values["bias"],
            col_values["bias"],
            embed_product
        ])
        # [None, 1]

    # prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        # calculate similarity
        with tf.name_scope("similarity"):
            # row similarity
            row_values = get_similarity(row_values, vocab_txt, k)
            # col similarity
            col_values = get_similarity(col_values, vocab_txt, k)

            embed_norm_product = tf.reduce_sum(tf.multiply(
                row_values["embed_norm"],
                col_values["embed_norm"]
            ), 1)

        predictions = {
            "predicted_value": predict_value,
            "row_id": row_values["id"],
            "row_embed": row_values["embed"],
            "row_bias": row_values["bias"],
            "col_id": col_values["id"],
            "col_embed": col_values["embed"],
            "col_bias": col_values["bias"],
            "embed_norm_product": embed_norm_product,
            "top_k_row_similarity": row_values["top_k_sim"],
            "top_k_row_string": row_values["top_k_string"],
            "top_k_col_similarity": col_values["top_k_sim"],
            "top_k_col_string": col_values["top_k_string"],
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # evaluation
    with tf.name_scope("losses"):
        mse_loss = tf.keras.losses.MeanSquaredError()(
            tf.expand_dims(labels[TARGET], -1), tf.expand_dims(predict_value, -1), sample_weights[TARGET]
        )
        # []
        loss = mse_loss + l2_reg * v1.losses.get_regularization_loss()
        # []
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    # training
    with tf.name_scope("train"):
        optimizer = v1.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=v1.train.get_or_create_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def get_estimator(job_dir, params):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=job_dir,
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
    estimator = get_estimator(params["job_dir"], params)

    # train spec
    dataset_args = {
        "file_pattern": params["train_csv"],
        "batch_size": params["batch_size"],
        **CONFIG["dataset_args"],
    }
    train_input_fn = get_keras_dataset_input_fn(**dataset_args, num_epochs=None)
    eval_input_fn = get_keras_dataset_input_fn(**dataset_args)

    # eval spec
    train_spec = get_train_spec(train_input_fn, params["train_steps"])
    exporter = get_exporter(get_serving_input_fn(**CONFIG["serving_input_fn_args"]))
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()

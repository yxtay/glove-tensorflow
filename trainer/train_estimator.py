import tensorflow as tf

from trainer.config import (
    CONFIG, EMBEDDING_SIZE, FEATURE_NAMES, L2_REG, LEARNING_RATE, OPTIMIZER, TARGET, TOP_K, VOCAB_TXT, WEIGHT,
)
from trainer.glove_utils import (
    MatrixFactorisation, cosine_similarity, get_id_string_table, get_string_id_table, init_params, parse_args,
)
from trainer.utils import (
    file_lines, get_csv_input_fn, get_estimator, get_eval_spec, get_exporter, get_optimizer, get_serving_input_fn,
    get_train_spec,
)

v1 = tf.compat.v1


def get_named_variables(model):
    variables = {
        "row_bias_layer": model.row_biases,
        "row_embedding_layer": model.row_embeddings,
        "col_bias_layer": model.col_biases,
        "col_embedding_layer": model.col_embeddings,
        "global_bias": model.global_bias,
        "row_biases": model.row_biases.weights[0],
        "row_embeddings": model.row_embeddings.weights[0],
        "col_biases": model.col_biases.weights[0],
        "col_embeddings": model.col_embeddings.weights[0],
    }
    return variables


def add_summary(model):
    with tf.name_scope("mf"):
        variables = get_named_variables(model)
        v1.summary.scalar("global_bias", variables["global_bias"])
        v1.summary.histogram("row_biases", variables["row_biases"])
        v1.summary.histogram("col_biases", variables["col_biases"])


def get_similarity(inputs, model, vocab_txt=VOCAB_TXT, top_k=TOP_K):
    # variables
    variables = get_named_variables(model)
    embedding_layer = variables["row_embedding_layer"]
    embeddings = embedding_layer.weights[0]
    # [vocab_size, embedding_size]

    # values
    token_id = inputs[0]
    # [None]
    embed = embedding_layer(token_id)
    # [None, embedding_size]
    cosine_sim = cosine_similarity(embed, embeddings)
    # [None, vocab_size]
    top_k_sim, top_k_idx = tf.math.top_k(cosine_sim, k=top_k, name="top_k_sim")
    # [None, top_k], [None, top_k]
    id_string_table = get_id_string_table(vocab_txt)
    top_k_string = id_string_table.lookup(tf.cast(top_k_idx, tf.int64), name="string_lookup")
    # [None, top_k]
    values = {
        "embed:": embed,
        "top_k_similarity": top_k_sim,
        "top_k_string": top_k_string,
    }
    return values


def model_fn(features, labels, mode, params):
    vocab_txt = params.get("vocab_txt", VOCAB_TXT)
    embedding_size = params.get("embedding_size", EMBEDDING_SIZE)
    l2_reg = params.get("l2_reg", L2_REG)
    optimizer_name = params.get("optimizer", OPTIMIZER)
    learning_rate = params.get("learning_rate", LEARNING_RATE)
    top_k = params.get("top_k", TOP_K)

    # features transform
    with tf.name_scope("features"):
        string_id_table = get_string_id_table(vocab_txt)
        inputs = [string_id_table.lookup(features[name], name=name + "_lookup") for name in FEATURE_NAMES]

    # model
    model = MatrixFactorisation(file_lines(vocab_txt), embedding_size, l2_reg)
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits = model(inputs, training=training)
    add_summary(model)

    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = get_similarity(inputs, model, vocab_txt, top_k)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # training
    optimizer = None
    if training:
        optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    # head
    head = tf.estimator.RegressionHead(weight_column=WEIGHT)
    return head.create_estimator_spec(
        features, mode, logits,
        labels=labels[TARGET],
        optimizer=optimizer,
        trainable_variables=model.trainable_variables,
        update_ops=model.get_updates_for(None) + model.get_updates_for(features),
        regularization_losses=model.get_losses_for(None) + model.get_losses_for(features),
    )


def main():
    args = parse_args()
    params = init_params(args.__dict__)

    # estimator
    estimator = get_estimator(model_fn, params)

    # input functions
    dataset_args = {
        "file_pattern": params["train_csv"],
        "batch_size": params["batch_size"],
        **CONFIG["input_fn_args"],
    }
    train_input_fn = get_csv_input_fn(**dataset_args, num_epochs=None)
    eval_input_fn = get_csv_input_fn(**dataset_args)

    # train, eval spec
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

import tensorflow as tf

from trainer.config import (
    CONFIG, EMBEDDING_SIZE, FEATURE_NAMES, L2_REG, LEARNING_RATE, OPTIMIZER, TARGET, TOP_K, VOCAB_TXT, WEIGHT,
)
from trainer.data_utils import get_csv_input_fn, get_serving_input_fn
from trainer.glove_utils import get_similarity, get_string_id_table, parse_args
from trainer.model_utils import MatrixFactorisation, add_summary, get_optimizer
from trainer.train_utils import get_estimator, get_eval_spec, get_exporter, get_train_spec
from trainer.utils import file_lines


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
    params = parse_args()

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

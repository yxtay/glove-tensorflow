import tensorflow as tf

from src.config import (
    COL_NAME, EMBEDDING_SIZE, L2_REG, LEARNING_RATE, OPTIMIZER, ROW_NAME, TARGET_NAME, TOP_K, VOCAB_TXT, WEIGHT_NAME,
)
from src.models.config_utils import parse_args
from src.models.data_utils import get_csv_input_fn, get_serving_input_fn
from src.models.model_utils import MatrixFactorisation, add_summary, get_predictions, get_string_id_table
from src.models.train_utils import get_estimator, get_eval_spec, get_exporter, get_optimizer, get_train_spec
from src.models.utils import file_lines


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

    # features transform
    with tf.name_scope("features"):
        string_id_table = get_string_id_table(vocab_txt)
        inputs = [string_id_table.lookup(features[name], name=name + "_lookup") for name in [row_name, col_name]]

    # model
    model = MatrixFactorisation(file_lines(vocab_txt), embedding_size, l2_reg)
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits = model(inputs, training=training)
    add_summary(model)

    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = get_predictions(inputs, model, vocab_txt, top_k)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # training
    optimizer = None
    if training:
        optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    # head
    head = tf.estimator.RegressionHead(weight_column=weight_name)
    return head.create_estimator_spec(
        features, mode, logits,
        labels=labels[target_name],
        optimizer=optimizer,
        trainable_variables=model.trainable_variables,
        update_ops=model.get_updates_for(None) + model.get_updates_for(features),
        regularization_losses=model.get_losses_for(None) + model.get_losses_for(features),
    )


def get_predict_input_fn(params):
    def arrange_input(line):
        return {name: line for name in [params["row_name"], params["col_name"]]}

    def input_fn():
        dataset = tf.data.TextLineDataset(params["vocab_txt"])
        dataset = dataset.map(arrange_input)
        dataset = dataset.batch(1)
        return dataset

    return input_fn


def estimator_predict(params):
    estimator = get_estimator(model_fn, params)
    predict_input_fn = get_predict_input_fn(params)
    predictions = estimator.predict(predict_input_fn)
    return predictions


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

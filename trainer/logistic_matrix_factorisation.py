import tensorflow as tf

from trainer.config import (
    COL_NAME, EMBEDDING_SIZE, L2_REG, LEARNING_RATE, NEG_FACTOR, NEG_NAME, OPTIMIZER, POS_NAME, ROW_NAME, TOP_K,
    VOCAB_TXT, parse_args, save_params,
)
from trainer.data_utils import get_csv_input_fn, get_serving_input_fn
from trainer.model_utils import MatrixFactorisation, add_summary, get_predictions, get_string_id_table
from trainer.train_utils import get_estimator, get_eval_spec, get_exporter, get_optimizer, get_train_spec
from trainer.utils import file_lines


def model_fn(features, labels, mode, params):
    row_name = params.get("row_name", ROW_NAME)
    col_name = params.get("col_name", COL_NAME)
    pos_name = params.get("pos_name", POS_NAME)
    neg_name = params.get("neg_name", NEG_NAME)
    vocab_txt = params.get("vocab_txt", VOCAB_TXT)
    embedding_size = params.get("embedding_size", EMBEDDING_SIZE)
    l2_reg = params.get("l2_reg", L2_REG)
    neg_factor = params.get("neg_factor", NEG_FACTOR)
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
    labels = {"pos": tf.ones_like(logits), "neg": tf.zeros_like(logits)}
    logits = {"pos": logits, "neg": logits}
    pos_head = tf.estimator.BinaryClassHead(weight_column=pos_name, name="pos")
    neg_head = tf.estimator.BinaryClassHead(weight_column=neg_name, name="neg")
    head = tf.estimator.MultiHead([pos_head, neg_head], [1, neg_factor])
    return head.create_estimator_spec(
        features, mode, logits,
        labels=labels,
        optimizer=optimizer,
        trainable_variables=model.trainable_variables,
        update_ops=model.get_updates_for(None) + model.get_updates_for(features),
        regularization_losses=model.get_losses_for(None) + model.get_losses_for(features),
    )


def main():
    params = parse_args()
    params["input_fn_args"].update({
        "select_columns": [params["row_name"], params["col_name"], params["pos_name"], params["neg_name"]],
        "target_names": [],
    })
    save_params(params)

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

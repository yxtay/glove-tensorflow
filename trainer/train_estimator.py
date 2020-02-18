import tensorflow as tf

from trainer.config import (
    CONFIG, EMBEDDING_SIZE, FEATURE_NAMES, L2_REG, LEARNING_RATE, OPTIMIZER, TARGET, VOCAB_TXT, WEIGHT,
)
from trainer.glove_utils import build_glove_model, get_string_id_table, init_params, parse_args
from trainer.utils import (
    get_csv_input_fn, get_eval_spec, get_exporter, get_loss_fn, get_minimise_op, get_optimizer, get_run_config,
    get_serving_input_fn, get_train_spec,
)

v1 = tf.compat.v1


def add_summary(model):
    glove_mf = model.get_layer("glove_value")
    v1.summary.scalar("mf/global_bias", model.weights[0])
    v1.summary.histogram("mf/row_biases", glove_mf.row_biases.weights[0])
    v1.summary.histogram("mf/col_biases", glove_mf.col_biases.weights[0])


def model_fn(features, labels, mode, params):
    vocab_txt = params.get("vocab_txt", VOCAB_TXT)
    embedding_size = params.get("embedding_size", EMBEDDING_SIZE)
    l2_reg = params.get("l2_reg", L2_REG)
    optimizer_name = params.get("optimizer", OPTIMIZER)
    learning_rate = params.get("learning_rate", LEARNING_RATE)

    with tf.name_scope("features"):
        string_id_table = get_string_id_table(vocab_txt)
        inputs = {name: string_id_table.lookup(features[name], name=name + "_lookup") for name in FEATURE_NAMES}

    model = build_glove_model(vocab_txt, embedding_size, l2_reg)
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    logit = model(inputs, training=training)
    add_summary(model)

    # head
    # head = tf.estimator.RegressionHead(weight_column=TARGET)
    # optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
    # return head.create_estimator_spec(
    #     features, mode, logits,
    #     labels=labels[TARGET],
    #     optimizer=optimizer,
    #     trainable_variables=model.trainable_variables,
    #     update_ops=model.get_updates_for(None) + model.get_updates_for(features),
    #     regularization_losses=model.get_losses_for(None) + model.get_losses_for(features),
    # )
    # prediction
    predictions = {"logit": logit}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # evaluation
    with tf.name_scope("losses"):
        mse_loss = get_loss_fn("MeanSquaredError")(
            labels[TARGET], tf.expand_dims(logit, -1), features.get(WEIGHT),
        )
        # []
        reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
        regularization_loss = tf.math.add_n(reg_losses)
        loss = mse_loss + regularization_loss
        # []
        v1.summary.scalar("error_loss", mse_loss)
        v1.summary.scalar("regularization_loss", regularization_loss)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss)

    # training
    optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
    minimise_op = get_minimise_op(loss, optimizer, model.trainable_variables)
    update_ops = model.get_updates_for(None) + model.get_updates_for(features)
    train_op = tf.group(*minimise_op, *update_ops, name="train_op")
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def get_estimator(job_dir, params):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=job_dir,
        config=get_run_config(),
        params=params
    )
    return estimator


def main():
    args = parse_args()
    params = init_params(args.__dict__)

    # estimator
    estimator = get_estimator(params["job_dir"], params)

    # input functions
    dataset_args = {
        "file_pattern": params["train_csv"],
        "batch_size": params["batch_size"],
        **CONFIG["dataset_args"],
    }
    train_input_fn = get_csv_input_fn(**dataset_args, num_epochs=None)
    eval_input_fn = get_csv_input_fn(**dataset_args)

    # train, eval spec
    train_spec = get_train_spec(train_input_fn, params["train_steps"])
    exporter = get_exporter(get_serving_input_fn(**CONFIG["serving_input_fn_args"]))
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()

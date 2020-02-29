import os

import tensorflow as tf

EVAL_INTERVAL = 300


def get_loss_fn(loss_name, **kwargs):
    loss_fn = tf.keras.losses.get({"class_name": loss_name, "config": kwargs})
    return loss_fn


def get_optimizer(optimizer_name="Adam", **kwargs):
    optimizer_config = {"class_name": optimizer_name, "config": kwargs}
    optimizer = tf.keras.optimizers.get(optimizer_config)
    return optimizer


def get_minimise_op(loss, optimizer, trainable_variables):
    with tf.name_scope("train"):
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
        minimise_op = optimizer.get_updates(loss, trainable_variables)
    return minimise_op


def get_run_config(save_checkpoints_secs=EVAL_INTERVAL):
    return tf.estimator.RunConfig(save_checkpoints_secs=min(save_checkpoints_secs, 300))


def get_estimator(model_fn, params):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params["job_dir"],
        config=get_run_config(),
        params=params,
    )


def get_train_spec(input_fn, train_steps):
    return tf.estimator.TrainSpec(input_fn=input_fn, max_steps=train_steps)


def get_exporter(serving_input_fn):
    return tf.estimator.LatestExporter(name="exporter", serving_input_receiver_fn=serving_input_fn)


def get_eval_spec(input_fn, exporter, throttle_secs=EVAL_INTERVAL):
    return tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=None,  # if None, until OutOfRangeError from input_fn
        exporters=exporter,
        start_delay_secs=min(throttle_secs, 120),
        throttle_secs=throttle_secs,
    )


def get_keras_estimator(keras_model, model_dir):
    return tf.keras.estimator.model_to_estimator(
        keras_model=keras_model,
        model_dir=model_dir,
        config=get_run_config(),
    )


def get_keras_callbacks(job_dir, model_pattern="model_{epoch:06d}", log_csv="log.csv"):
    log_csv = os.path.join(job_dir, log_csv)
    model_path = os.path.join(job_dir, model_pattern)
    callbacks = [
        tf.keras.callbacks.CSVLogger(log_csv),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path),
        tf.keras.callbacks.TensorBoard(log_dir=job_dir),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks

from argparse import ArgumentParser

import tensorflow as tf

from trainer.config import (
    BATCH_SIZE, EMBEDDING_SIZE, FEATURE_NAMES, L2_REG, LEARNING_RATE, OPTIMIZER_NAME, TRAIN_CSV, TRAIN_STEPS,
    VOCAB_TXT,
)
from trainer.glove_utils import build_glove_model, get_glove_dataset, get_lookup_tables, init_params
from trainer.utils import (
    get_eval_spec, get_exporter, get_keras_dataset_input_fn, get_minimise_op, get_optimizer,
    get_run_config, get_train_spec,
)

fc = tf.feature_column


def get_feature_columns(vocab_txt, embedding_size=EMBEDDING_SIZE):
    cat_fc = [fc.categorical_column_with_vocabulary_file(key, vocab_txt, default_value=0)
              for key in FEATURE_NAMES]
    emb_fc = [fc.embedding_column(col, embedding_size) for col in cat_fc]
    return {
        "categorical": cat_fc,
        "embedding": emb_fc,
    }


def model_fn(features, labels, mode, params):
    vocab_txt = params.get("vocab_txt", VOCAB_TXT)
    embedding_size = params.get("embedding_size", EMBEDDING_SIZE)
    optimizer_name = params.get("optimizer", OPTIMIZER_NAME)
    learning_rate = params.get("learning_rate", LEARNING_RATE)
    l2_reg = params.get("l2_reg", L2_REG)

    if set(features.keys()) == {"features", "sample_weights"}:
        sample_weights = features["sample_weights"]
        features = features["features"]

    model = build_glove_model(vocab_txt, embedding_size, l2_reg)

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    predict_value = model(features, training=training)

    # prediction
    predictions = {"predict_value": predict_value}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # evaluation
    with tf.name_scope("losses"):
        mse_loss = tf.keras.losses.MeanSquaredError()(
            labels["glove_value"],
            tf.expand_dims(predict_value, -1),
            sample_weights["glove_value"],
        )
        # []
        reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
        loss = tf.math.add_n([mse_loss] + reg_losses)
        # []
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss)

    # training
    optimizer = get_optimizer(optimizer_name, learning_rate)
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


def get_serving_input_fn():
    def serving_input_fn():
        features = {
            key: tf.compat.v1.placeholder(tf.int32, [None], name=key)
            for key in FEATURE_NAMES
        }
        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=features,
        )

    return serving_input_fn


def main(args):
    params = init_params(args.__dict__)

    # estimator
    estimator = get_estimator(params["job_dir"], params)

    # input functions
    dataset_args = {
        "dataset_fn": get_glove_dataset,
        "file_pattern": params["train_csv"],
        "vocab_txt": params["vocab_txt"],
        "batch_size": params["batch_size"],
    }
    train_input_fn = get_keras_dataset_input_fn(**dataset_args, num_epochs=None)
    eval_input_fn = get_keras_dataset_input_fn(**dataset_args)

    # train, eval spec
    train_spec = get_train_spec(train_input_fn, params["train_steps"])
    exporter = get_exporter(get_serving_input_fn())
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default=TRAIN_CSV,
        help="path to the training csv data (default: %(default)s)"
    )
    parser.add_argument(
        "--vocab-txt",
        default=VOCAB_TXT,
        help="path to the vocab txt (default: %(default)s)"
    )
    parser.add_argument(
        "--job-dir",
        default="checkpoints/glove",
        help="job directory (default: %(default)s)"
    )
    parser.add_argument(
        "--use-job-dir-path",
        action="store_true",
        help="flag whether to use raw job_dir path (default: %(default)s)"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=EMBEDDING_SIZE,
        help="embedding size (default: %(default)s)"
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=L2_REG,
        help="scale of l2 regularisation (default: %(default)s)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="learning rate (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=TRAIN_STEPS,
        help="number of training steps (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args)

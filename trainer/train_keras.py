import os
from argparse import ArgumentParser

import tensorflow as tf

from trainer.config import (
    BATCH_SIZE, EMBEDDING_SIZE, L2_REG, LEARNING_RATE, STEPS_PER_EPOCH, TRAIN_CSV, TRAIN_STEPS, VOCAB_TXT,
)
from trainer.glove_utils import build_glove_model, get_glove_dataset, init_params
from trainer.utils import get_keras_callbacks

fc = tf.feature_column


def main(args):
    params = init_params(args.__dict__)
    job_dir = params["job_dir"]

    # set up model and compile
    model = build_glove_model(params["vocab_txt"], params["embedding_size"], params["l2_reg"])
    model.compile(optimizer=tf.keras.optimizers.Adamax(params["learning_rate"]),
                  loss=tf.keras.losses.MeanSquaredError())

    # set up train, validation dataset
    dataset_args = {
        "file_pattern": params["train_csv"],
        "vocab_txt": params["vocab_txt"],
        "batch_size": params["batch_size"],
    }
    train_dataset = get_glove_dataset(**dataset_args, num_epochs=None)
    validation_dataset = get_glove_dataset(**dataset_args)

    # train and evaluate
    history = model.fit(
        train_dataset,
        epochs=params["train_steps"] // params["steps_per_epoch"],
        callbacks=get_keras_callbacks(job_dir),
        validation_data=validation_dataset,
        steps_per_epoch=params["steps_per_epoch"],
    )
    model_path = os.path.join(job_dir, "model")
    model.save(model_path)


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
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=STEPS_PER_EPOCH,
        help="number of steps per checkpoint (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args)

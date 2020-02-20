import json
import os
from argparse import ArgumentParser
from datetime import datetime

import tensorflow as tf

from trainer.config import (
    BATCH_SIZE, CONFIG, EMBEDDING_SIZE, FEATURE_NAMES, L2_REG, LEARNING_RATE, OPTIMIZER, STEPS_PER_EPOCH, TOP_K,
    TRAIN_CSV, TRAIN_STEPS, VOCAB_TXT,
)
from trainer.utils import file_lines, get_csv_dataset


def get_embedding_layer(vocab_size, embedding_size=EMBEDDING_SIZE, name="embedding", l2_reg=L2_REG):
    scaled_l2_reg = l2_reg / (vocab_size * embedding_size)
    regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=scaled_l2_reg)
    embedding_layer = tf.keras.layers.Embedding(
        vocab_size, embedding_size,
        embeddings_regularizer=regularizer,
        name=name
    )
    return embedding_layer


class MatrixFactorisation(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG, **kwargs):
        super(MatrixFactorisation, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg

    def build(self, input_shapes):
        self.row_embeddings = get_embedding_layer(
            self.vocab_size, self.embedding_size,
            name="row_embedding",
            l2_reg=self.l2_reg
        )
        self.row_biases = get_embedding_layer(
            self.vocab_size, 1,
            name="row_bias",
            l2_reg=self.l2_reg
        )

        self.col_embeddings = get_embedding_layer(
            self.vocab_size, self.embedding_size,
            name="col_embedding",
            l2_reg=self.l2_reg
        )
        self.col_biases = get_embedding_layer(
            self.vocab_size, 1,
            name="col_bias",
            l2_reg=self.l2_reg
        )

        regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=L2_REG)
        self.global_bias = self.add_weight(
            name="global_bias",
            initializer="zeros",
            regularizer=regularizer,
        )

    def call(self, inputs):
        row_id, col_id = inputs

        row_embed = self.row_embeddings(row_id)
        row_bias = self.row_biases(row_id)

        col_embed = self.col_embeddings(col_id)
        col_bias = self.col_biases(col_id)

        embed_product = tf.keras.layers.dot([row_embed, col_embed], axes=-1, name="embed_product")
        global_bias = tf.ones_like(embed_product) * self.global_bias
        logits = tf.keras.layers.Add(name="logits")([embed_product, row_bias, col_bias, global_bias])
        return logits

    def get_config(self):
        config = super(MatrixFactorisation, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "l2_reg": self.l2_reg,
        })
        return config


def build_glove_model(vocab_txt=VOCAB_TXT, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG):
    # init layers
    mf_layer = MatrixFactorisation(file_lines(vocab_txt), embedding_size, l2_reg, name="glove_value")

    # build model
    inputs = [tf.keras.Input((), name=name) for name in FEATURE_NAMES]
    glove_value = mf_layer(inputs)
    glove_model = tf.keras.Model(inputs, glove_value, name="glove_model")
    return glove_model


class GloVeModel(tf.keras.Model):
    def __init__(self, vocab_txt=VOCAB_TXT, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG, **kwargs):
        super(GloVeModel, self).__init__(**kwargs)
        self.vocab_txt = vocab_txt
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg

    def build(self, input_shapes):
        self.mf_layer = MatrixFactorisation(
            file_lines(self.vocab_txt),
            self.embedding_size,
            self.l2_reg,
            name="glove_value"
        )

    def call(self, inputs):
        glove_value = self.mf_layer(inputs)
        return glove_value


def get_string_id_table(vocab_txt=VOCAB_TXT):
    lookup_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        vocab_txt,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
    ), 0, name="string_id_table")
    return lookup_table


def get_id_string_table(vocab_txt=VOCAB_TXT):
    lookup_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        vocab_txt,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
    ), "<UNK>", name="id_string_table")
    return lookup_table


def get_glove_dataset(file_pattern=TRAIN_CSV, vocab_txt=VOCAB_TXT, batch_size=BATCH_SIZE, num_epochs=1):
    string_id_table = get_string_id_table(vocab_txt)

    def lookup(features, targets, weights):
        features = {name: string_id_table.lookup(values, name=name + "_lookup") for name, values in features.items()}
        return features, targets, weights

    dataset_args = {
        "file_pattern": file_pattern,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        **CONFIG["dataset_args"],
    }
    dataset = get_csv_dataset(**dataset_args).map(lookup, num_parallel_calls=-1)
    return dataset


def init_params(params):
    # job_dir
    if not params["use_job_dir_path"]:
        datetime_now = datetime.now()
        job_dir = "{job_dir}_{datetime:%Y%m%d_%H%M%S}".format(job_dir=params["job_dir"], datetime=datetime_now)
        params["job_dir"] = job_dir
    tf.io.gfile.makedirs(job_dir)

    # vocab_txt
    output_vocab_txt = os.path.join(job_dir, os.path.basename(params["vocab_txt"]))
    tf.io.gfile.copy(params["vocab_txt"], output_vocab_txt, overwrite=True)
    params["vocab_txt"] = output_vocab_txt

    # save params
    params_json = os.path.join(job_dir, "params.json")
    with tf.io.gfile.GFile(params_json, "w") as f:
        json.dump(params, f, indent=2)
    return params


def parse_args():
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
        "--optimizer",
        default=OPTIMIZER,
        help="name of optimzer (default: %(default)s)"
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="number of similar token (default: %(default)s)"
    )
    args = parser.parse_args()
    return args

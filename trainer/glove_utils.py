import json
import os
from datetime import datetime

import tensorflow as tf

from trainer.config import BATCH_SIZE, CONFIG, EMBEDDING_SIZE, FEATURE_NAMES, L2_REG, TRAIN_CSV, VOCAB_TXT
from trainer.utils import file_lines, get_csv_dataset

fc = tf.feature_column


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
            shape=(1,),
            regularizer=regularizer,
            trainable=True
        )

    def call(self, inputs):
        row_id, col_id = inputs

        row_embed = self.row_embeddings(row_id)
        row_bias = self.row_biases(row_id)

        col_embed = self.col_embeddings(col_id)
        col_bias = self.col_biases(col_id)

        embed_product = tf.keras.layers.dot([row_embed, col_embed], axes=-1, name="embed_product")
        global_bias = tf.ones_like(embed_product) * self.global_bias
        logit = tf.keras.layers.Add(name="logit")([embed_product, row_bias, col_bias, global_bias])
        return logit

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "l2_reg": self.l2_reg
        }
        return config


def build_glove_model(vocab_txt, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG):
    vocab_size = file_lines(vocab_txt)
    mf_layer = MatrixFactorisation(
        vocab_size, embedding_size, l2_reg,
        name="glove_value"
    )

    inputs = [tf.keras.Input((), name=name) for name in FEATURE_NAMES]
    glove_value = mf_layer(inputs)
    glove_model = tf.keras.Model(inputs, glove_value, name="glove_model")
    return glove_model


class GloVeModelWithFC(tf.keras.Model):
    def __init__(self, vocab_txt=VOCAB_TXT, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG, **kwargs):
        super(GloVeModelWithFC, self).__init__(**kwargs)
        self.vocab_txt = vocab_txt
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg

    def build(self, input_shapes):
        cat_fc = [fc.categorical_column_with_vocabulary_file(key, self.vocab_txt, default_value=0)
                  for key in FEATURE_NAMES]
        self.row_fc, self.col_fc = [tf.keras.layers.DenseFeatures(fc.indicator_column(col), name=col.name)
                                    for col in cat_fc]
        self.mf_layer = MatrixFactorisation(
            file_lines(self.vocab_txt),
            self.embedding_size,
            self.l2_reg,
            name="glove_value"
        )

    def call(self, inputs):
        row_id = self.row_fc(inputs)
        col_id = self.col_fc(inputs)
        glove_value = self.mf_layer([row_id, col_id])
        return glove_value


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


def get_lookup_tables(vocab_txt=VOCAB_TXT):
    string_id_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        vocab_txt,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
    ), 0, name="string_id_table")

    id_string_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        vocab_txt,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
    ), "<UNK>", name="id_string_table")
    return {
        "string_id_table": string_id_table,
        "id_string_table": id_string_table,
    }


def get_glove_dataset(file_pattern=TRAIN_CSV, vocab_txt=VOCAB_TXT, batch_size=BATCH_SIZE, num_epochs=1):
    string_id_table = get_lookup_tables(vocab_txt)["string_id_table"]

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

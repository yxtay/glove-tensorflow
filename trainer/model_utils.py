import tensorflow as tf

from trainer.config import VOCAB_TXT, TOP_K
from trainer.data_utils import get_csv_dataset
from trainer.utils import cosine_similarity


def get_embedding_layer(vocab_size, embedding_size, l2_reg=0.01, name="embedding"):
    scaled_l2_reg = l2_reg / (vocab_size * embedding_size)
    regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=scaled_l2_reg)
    embedding_layer = tf.keras.layers.Embedding(
        vocab_size, embedding_size,
        embeddings_regularizer=regularizer,
        name=name
    )
    return embedding_layer


class MatrixFactorisation(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, l2_reg=0.01, name="matrix_factorisation", **kwargs):
        super(MatrixFactorisation, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg

    def build(self, input_shapes):
        self.row_embeddings = get_embedding_layer(self.vocab_size, self.embedding_size, self.l2_reg, "row_embedding")
        self.row_biases = get_embedding_layer(self.vocab_size, 1, self.l2_reg, "row_bias")

        self.col_embeddings = get_embedding_layer(self.vocab_size, self.embedding_size, self.l2_reg, "col_embedding")
        self.col_biases = get_embedding_layer(self.vocab_size, 1, self.l2_reg, "col_bias")

        regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=self.l2_reg)
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


def add_summary(model):
    with tf.name_scope("mf"):
        variables = get_named_variables(model)
        tf.compat.v1.summary.scalar("global_bias", variables["global_bias"])
        tf.compat.v1.summary.histogram("row_biases", variables["row_biases"])
        tf.compat.v1.summary.histogram("col_biases", variables["col_biases"])


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


def get_glove_dataset(vocab_txt=VOCAB_TXT, **kwargs):
    string_id_table = get_string_id_table(vocab_txt)

    def lookup(features, targets, weights):
        features = {name: string_id_table.lookup(features[name], name=name + "_lookup")
                    for name in kwargs["feature_names"]}
        return features, targets, weights

    dataset = get_csv_dataset(**kwargs).map(lookup, num_parallel_calls=-1)
    return dataset

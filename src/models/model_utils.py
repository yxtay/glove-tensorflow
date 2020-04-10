import tensorflow as tf

from src.config import EMBEDDING_SIZE, L2_REG, TOP_K, VOCAB_TXT
from src.models.utils import cosine_similarity


def get_embedding_layer(vocab_size, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG, name="embedding"):
    scaled_l2_reg = l2_reg / embedding_size
    regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=scaled_l2_reg)
    embedding_layer = tf.keras.layers.Embedding(
        vocab_size, embedding_size,
        activity_regularizer=regularizer,
        name=name
    )
    return embedding_layer


def compute_activity_loss(tensor, regularizer):
    batch_activty_loss = regularizer(tensor)
    mean_activity_loss = batch_activty_loss / tf.cast(tf.shape(tensor)[0], batch_activty_loss.dtype)
    return mean_activity_loss


class MatrixFactorisation(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG, name="matrix_factorisation", **kwargs):
        super(MatrixFactorisation, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg

    def build(self, input_shapes):
        self.row_embeddings = get_embedding_layer(self.vocab_size, self.embedding_size, self.l2_reg, "row_embedding")
        self.row_biases = get_embedding_layer(self.vocab_size, 1, self.l2_reg, "row_bias")

        self.col_embeddings = get_embedding_layer(self.vocab_size, self.embedding_size, self.l2_reg, "col_embedding")
        self.col_biases = get_embedding_layer(self.vocab_size, 1, self.l2_reg, "col_bias")

        self.regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=self.l2_reg)
        self.global_bias = self.add_weight(name="global_bias", initializer="zeros")

    def call(self, inputs):
        row_id, col_id = inputs

        row_embed = self.row_embeddings(row_id)
        row_bias = self.row_biases(row_id)

        col_embed = self.col_embeddings(col_id)
        col_bias = self.col_biases(col_id)

        embed_product = tf.keras.layers.dot([row_embed, col_embed], axes=-1, name="embed_product")
        global_bias = tf.ones_like(embed_product) * self.global_bias
        self.add_loss(compute_activity_loss(global_bias, self.regularizer))
        logits = tf.keras.layers.Add(name="logits")([embed_product, row_bias, col_bias, global_bias])
        return logits

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "l2_reg": self.l2_reg,
        }
        config.update(super(MatrixFactorisation, self).get_config())
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


def get_predictions(inputs, model, vocab_txt=VOCAB_TXT, top_k=TOP_K):
    with tf.name_scope("predictions"):
        id_string_table = get_id_string_table(vocab_txt)
        # variables
        variables = get_named_variables(model)
        embedding_layer = variables["row_embedding_layer"]
        embeddings = embedding_layer.weights[0]
        # [vocab_size, embedding_size]

        # values
        input_id = inputs[0]
        # [None]
        input_string = id_string_table.lookup(input_id, name="input_string_lookup")
        # [None]
        input_embedding = embedding_layer(input_id)
        # [None, embedding_size]
        cosine_sim = cosine_similarity(input_embedding, embeddings)
        # [None, vocab_size]
        top_k_sim, top_k_idx = tf.math.top_k(cosine_sim, k=top_k, name="top_k_sim")
        # [None, top_k], [None, top_k]
        id_string_table = get_id_string_table(vocab_txt)
        top_k_string = id_string_table.lookup(tf.cast(top_k_idx, tf.int64), name="top_k_string_lookup")
        # [None, top_k]
    values = {
        "input_string": input_string,
        "input_embedding": input_embedding,
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

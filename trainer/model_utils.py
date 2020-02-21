import tensorflow as tf


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


def add_summary(model):
    with tf.name_scope("mf"):
        variables = get_named_variables(model)
        tf.compat.v1.summary.scalar("global_bias", variables["global_bias"])
        tf.compat.v1.summary.histogram("row_biases", variables["row_biases"])
        tf.compat.v1.summary.histogram("col_biases", variables["col_biases"])


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

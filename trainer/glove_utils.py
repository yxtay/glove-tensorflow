import tensorflow as tf

from trainer.config import EMBEDDING_SIZE, L2_REG, ROW_COL_NAMES, TOP_K, VOCAB_TXT
from trainer.data_utils import get_csv_dataset
from trainer.model_utils import MatrixFactorisation, get_named_variables
from trainer.utils import cosine_similarity


def build_glove_model(vocab_size, embedding_size=EMBEDDING_SIZE, l2_reg=L2_REG, row_col_names=ROW_COL_NAMES):
    # init layers
    mf_layer = MatrixFactorisation(vocab_size, embedding_size, l2_reg, name="glove_value")

    # build model
    inputs = [tf.keras.Input((), name=name) for name in row_col_names]
    glove_value = mf_layer(inputs)
    glove_model = tf.keras.Model(inputs, glove_value, name="glove_model")
    return glove_model


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

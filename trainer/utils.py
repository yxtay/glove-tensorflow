import tensorflow as tf


def file_lines(fname):
    i = -1
    with tf.io.gfile.GFile(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def cosine_similarity(a, b):
    a_norm = tf.math.l2_normalize(a, -1)
    # [None, embedding_size]
    b_norm = tf.math.l2_normalize(b, -1)
    # [vocab_size, embedding_size]
    cosine_sim = tf.matmul(a_norm, b_norm, transpose_b=True)
    # [None, vocab_size]
    return cosine_sim

import tensorflow as tf


def get_optimizer(optimizer_name="Adam", learning_rate=0.001):
    optimizer_classes = {
        "Adagrad": tf.train.AdagradOptimizer,
        "Adam": tf.train.AdamOptimizer,
        "Ftrl": tf.train.FtrlOptimizer,
        "RMSProp": tf.train.RMSPropOptimizer,
        "SGD": tf.train.GradientDescentOptimizer,
    }
    optimizer = optimizer_classes[optimizer_name](learning_rate=learning_rate)
    return optimizer


def get_train_op(loss, optimizer):
    with tf.name_scope("train"):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return train_op

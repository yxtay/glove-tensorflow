import glob

import tensorflow as tf

EVAL_INTERVAL = 300


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


def get_input_fn(path_pattern, col_names, col_defaults, label_col,
                 mode=tf.estimator.ModeKeys.TRAIN, batch_size=32):
    def input_fn():
        def parse_csv(value):
            columns = tf.decode_csv(value, col_defaults)
            features = dict(zip(col_names, columns))
            label = features[label_col]
            return features, label

        with tf.name_scope("input_fn"):
            # read, parse, shuffle and batch dataset
            file_paths = glob.glob(path_pattern, recursive=True)
            dataset = tf.data.TextLineDataset(file_paths).skip(1)  # skip header
            if mode == tf.estimator.ModeKeys.TRAIN:
                # shuffle and repeat
                dataset = dataset.shuffle(16 * batch_size).repeat()

            dataset = dataset.map(parse_csv, num_parallel_calls=8)
            dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


def get_serving_input_fn(numeric_features=(), string_features=()):
    def serving_input_fn():
        features = {
            key: tf.placeholder(tf.float32, [None], name=key)
            for key in numeric_features
        }
        features.update({
            key: tf.placeholder(tf.string, [None], name=key)
            for key in string_features
        })
        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=features
        )

    return serving_input_fn


def get_run_config():
    return tf.estimator.RunConfig(
        save_checkpoints_secs=EVAL_INTERVAL,
        keep_checkpoint_max=5
    )


def get_train_spec(input_fn, train_steps):
    return tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=train_steps
    )


def get_exporter(serving_input_fn):
    return tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=serving_input_fn
    )


def get_eval_spec(input_fn, exporter):
    return tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=None,  # until OutOfRangeError from input_fn
        exporters=exporter,
        start_delay_secs=min(EVAL_INTERVAL, 120),
        throttle_secs=EVAL_INTERVAL
    )

import numpy as np
import tensorflow as tf

EVAL_INTERVAL = 300


def get_column_info(df, dtypes={}):
    type_defaults = {"bool": ["false"], "int": [0], "float": [0.0], "str": [""]}

    col_names = df.columns.tolist()
    # get column type names
    col_types = []
    for col in col_names:
        if col in dtypes:
            col_types.append(dtypes[col])
        elif df[col].dtype in ("float", "float32", "float64"):
            col_types.append("float")
        elif df[col].dtype in ("int", "int32", "int64"):
            col_types.append("int")
        elif df[col].dtype in ("bool", "bool8"):
            col_types.append("bool")
        elif df[col].dtype in ("object", "str"):
            col_types.append("str")
        else:
            col_types.append("str")
    # get column defaults
    col_defaults = [type_defaults.get(t, [""]) for t in col_types]
    return {
        "col_names": col_names,
        "col_types": col_types,
        "col_defaults": col_defaults
    }


def get_bucket_size(df, cols, min_bucket_size=10, max_bucket_size=None):
    # count number of categories
    nunique = np.array([df[col].nunique_approx().compute() for col in cols])
    # find order of categories
    order = np.floor(np.log10(nunique))
    # find multiple for order of categories
    multiple = np.ceil(nunique / np.power(10, order))
    # compute bucket size
    bucket_size = np.clip(multiple * np.power(10, order), min_bucket_size, max_bucket_size).astype(np.int)
    return dict(zip(cols, bucket_size))


def get_feature_columns(numeric_features=[], numeric_bucketised={},
                        cat_vocab_list={}, cat_hash_bucket={},
                        embedding_size=4):
    # numeric
    numeric_fc = [tf.feature_column.numeric_column(col) for col in numeric_features]
    # numeric bucketised
    numeric_bucketised_fc = [tf.feature_column.bucketized_column(tf.feature_column.numeric_column(col), boundaries)
                             for col, boundaries in numeric_bucketised.items()]
    # categorical with vocabulary list
    cat_vocab_list_fc = [tf.feature_column.categorical_column_with_vocabulary_list(col, vocab, num_oov_buckets=1)
                         for col, vocab in cat_vocab_list.items()]
    # categorical with hash buckets
    cat_hash_bucket_fc = [tf.feature_column.categorical_column_with_hash_bucket(col, bucket_size)
                          for col, bucket_size in cat_hash_bucket.items()]
    # embedding
    cat_fc = numeric_bucketised_fc + cat_vocab_list_fc + cat_hash_bucket_fc
    embedding_fc = [tf.feature_column.embedding_column(fc, embedding_size) for fc in cat_fc]

    linear_fc = numeric_fc + cat_fc
    deep_fc = numeric_fc + embedding_fc
    return {
        "numeric": numeric_fc,
        "categorical": cat_fc,
        "embedding": embedding_fc,
        "linear": linear_fc,
        "deep": deep_fc,
    }


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
        def name_columns(*columns):
            features = dict(zip(col_names, columns))
            label = features[label_col]
            return features, label

        with tf.name_scope("input_fn"):
            # read, parse, shuffle and batch dataset
            file_paths = tf.gfile.Glob(path_pattern)
            dataset = tf.data.experimental.CsvDataset(file_paths, col_defaults, header=True)
            # repeat for train
            if mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.repeat()

            dataset = (dataset
                       .map(name_columns, num_parallel_calls=8)
                       .shuffle(16 * batch_size)
                       .batch(batch_size))
        return dataset

    return input_fn


def get_serving_input_fn(int_features=(), float_features=(), string_features=()):
    def serving_input_fn():
        features = {}
        features.update({
            key: tf.placeholder(tf.int32, [None], name=key)
            for key in int_features
        })
        features.update({
            key: tf.placeholder(tf.float32, [None], name=key)
            for key in float_features
        })
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

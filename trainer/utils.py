import os

import numpy as np
import pandas as pd
import tensorflow as tf

EVAL_INTERVAL = 300
EVAL_STEPS = None  # None, until OutOfRangeError from input_fn


def get_column_info(df, dtypes={}):
    type_defaults = {"bool": ["false"], "int": [0], "float": [0.0], "str": [""]}

    col_names = df.columns.tolist()
    # get column type names
    col_types = []
    for col in col_names:
        if col in dtypes:
            col_types.append(dtypes[col])
        elif pd.api.types.is_float_dtype(df[col]):
            col_types.append("float")
        elif pd.api.types.is_integer_dtype(df[col]):
            col_types.append("int")
        elif pd.api.types.is_bool_dtype(df[col]):
            col_types.append("bool")
        elif pd.api.types.is_string_dtype(df[col]):
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


def get_csv_dataset(file_pattern, feature_names, target_names=(), weight_names=(),
                    batch_size=32, num_epochs=1, compression_type=""):
    def arrange_columns(features):
        output = features

        if len(target_names) > 0:
            targets = {col: features.pop(col) for col in target_names}
            output = features, targets

            if len(weight_names) > 0:
                weights = {target_col: features.pop(weight_col)
                           for target_col, weight_col in zip(target_names, weight_names)}
                output = features, targets, weights

        return output

    select_columns = feature_names + target_names + weight_names
    with tf.name_scope("dataset"):
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            select_columns=select_columns,
            num_epochs=num_epochs,
            num_parallel_reads=-1,
            sloppy=True,
            num_rows_for_inference=100,
            compression_type=compression_type,
        )
        dataset = dataset.map(arrange_columns, num_parallel_calls=-1)
    return dataset


def get_csv_input_fn(file_pattern, feature_names, target_names=(), weight_names=(),
                     batch_size=32, num_epochs=1, compression_type=""):
    def arrange_columns(features):
        output = features

        if len(target_names) > 0:
            targets = tf.stack([features.pop(col) for col in target_names], -1)
            features["sample_weights"] = tf.stack([features.pop(col) for col in weight_names], -1)
            output = features, targets

        return output

    def input_fn():
        select_columns = feature_names + target_names + weight_names
        with tf.name_scope("input_fn"):
            dataset = tf.data.experimental.make_csv_dataset(
                file_pattern=file_pattern,
                batch_size=batch_size,
                select_columns=select_columns,
                num_epochs=num_epochs,
                num_parallel_reads=-1,
                sloppy=True,
                num_rows_for_inference=100,
                compression_type=compression_type,
            )
            dataset = dataset.map(arrange_columns, num_parallel_calls=-1)
        return dataset

    return input_fn


def get_keras_estimator_input_fn(dataset_fn=get_csv_dataset, **kwargs):
    def map_keras_model_to_estimator(*values):
        if len(values) == 3:
            features, targets, weights = values
            targets = tf.stack([targets[name] for name in targets], -1)
            weights = tf.stack([weights[name] for name in targets], -1)
            return {"features": features, "sample_weights": weights}, targets

        return values
    
    def input_fn():
        with tf.name_scope("input_fn"):
            dataset = dataset_fn(**kwargs)
            dataset = dataset.map(map_keras_model_to_estimator, num_parallel_calls=-1)
        return dataset

    return input_fn


def get_serving_input_fn(int_features=(), float_features=(), string_features=()):
    def serving_input_fn():
        features = {}
        features.update({
            key: tf.compat.v1.placeholder(tf.int32, [None], name=key)
            for key in int_features
        })
        features.update({
            key: tf.compat.v1.placeholder(tf.float32, [None], name=key)
            for key in float_features
        })
        features.update({
            key: tf.compat.v1.placeholder(tf.string, [None], name=key)
            for key in string_features
        })
        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=features
        )

    return serving_input_fn


def get_run_config(save_checkpoints_secs=EVAL_INTERVAL, keep_checkpoint_max=5):
    return tf.estimator.RunConfig(
        save_checkpoints_secs=min(save_checkpoints_secs, 300),
        keep_checkpoint_max=keep_checkpoint_max
    )


def get_train_spec(input_fn, train_steps):
    return tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=train_steps
    )


def get_exporter(serving_input_fn, exports_to_keep=5):
    return tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=serving_input_fn,
        exports_to_keep=exports_to_keep
    )


def get_eval_spec(input_fn, exporter, steps=EVAL_STEPS, throttle_secs=EVAL_INTERVAL):
    return tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=steps,
        exporters=exporter,
        start_delay_secs=min(throttle_secs, 120),
        throttle_secs=throttle_secs
    )


def file_lines(fname):
    i = -1
    with tf.io.gfile.GFile(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_keras_callbacks(job_dir, model_pattern="model_{epoch:06d}", log_csv="log.csv"):
    log_csv = os.path.join(job_dir, log_csv)
    model_path = os.path.join(job_dir, model_pattern)
    callbacks = [
        tf.keras.callbacks.CSVLogger(log_csv),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path),
        tf.keras.callbacks.TensorBoard(log_dir=job_dir),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks

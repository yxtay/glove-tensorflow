import numpy as np
import pandas as pd
import tensorflow as tf


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
    col_info = {"col_names": col_names, "col_types": col_types, "col_defaults": col_defaults}
    return col_info


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


def get_csv_input_fn(file_pattern, select_columns=None, target_names=[],
                     batch_size=32, num_epochs=1, **kwargs):
    def arrange_columns(features):
        targets = {col: features.pop(col) for col in target_names}
        return features, targets

    def input_fn():
        with tf.name_scope("input_fn"):
            dataset = tf.data.experimental.make_csv_dataset(
                file_pattern=file_pattern,
                batch_size=batch_size,
                select_columns=select_columns,
                num_epochs=num_epochs,
                num_parallel_reads=8,
                sloppy=True,  # improves performance, non-deterministic ordering
                num_rows_for_inference=100,  # if None, read all the rows
                **kwargs,
            )
            if len(target_names) > 0:
                dataset = dataset.map(arrange_columns, num_parallel_calls=-1)
        return dataset

    return input_fn


def get_csv_dataset(file_pattern, select_columns=None, target_names=[], weight_names=[],
                    batch_size=32, num_epochs=1, **kwargs):
    def arrange_columns(features, targets):
        targets = {col: tf.expand_dims(targets[col], -1) for col in target_names}
        weights = {target_col: features.pop(weight_col) for target_col, weight_col in
                   zip(target_names, weight_names)}
        output = features, targets, weights

        return output

    with tf.name_scope("dataset"):
        input_fn = get_csv_input_fn(file_pattern, select_columns, target_names, batch_size, num_epochs, **kwargs)
        dataset = input_fn()
        if len(weight_names) > 0:
            dataset = dataset.map(arrange_columns, num_parallel_calls=-1)
    return dataset


def get_keras_estimator_input_fn(dataset_fn=get_csv_dataset, **kwargs):
    def map_keras_model_to_estimator(*values):
        if len(values) == 3:
            features, targets, weights = values
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

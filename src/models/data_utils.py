import tensorflow as tf


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
                num_epochs=num_epochs,  # if None, repeat forever
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

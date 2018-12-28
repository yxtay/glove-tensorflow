import tensorflow as tf

COLUMNS = ["row_token_id", "column_token_id", "interaction",
           "row_token", "column_token", "glove_weight", "glove_value"]
DEFAULTS = [[0], [0], [0.0], ["null"], ["null"], [0.0], [0.0]]
LABEL_COL = "glove_value"


def get_input_fn(csv_path, mode=tf.estimator.ModeKeys.TRAIN, batch_size=32):
    def input_fn():
        def parse_csv(value):
            columns = tf.decode_csv(value, DEFAULTS)
            features = dict(zip(COLUMNS, columns))
            label = features[LABEL_COL]
            return features, label

        # read, parse, shuffle and batch dataset
        dataset = tf.data.TextLineDataset(csv_path).skip(1)  # skip header
        if mode == tf.estimator.ModeKeys.TRAIN:
            # shuffle and repeat
            dataset = dataset.shuffle(16 * batch_size).repeat()

        dataset = dataset.map(parse_csv, num_parallel_calls=8)
        dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


def get_serving_input_fn(string_features=("row_token", "column_token")):
    def serving_input_fn():
        features = {
            key: tf.placeholder(tf.string, [None], name=key)
            for key in string_features
        }
        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=features
        )

    return serving_input_fn

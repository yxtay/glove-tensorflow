import tensorflow as tf

COLUMNS = ["row_id", "column_id", "interaction", "row_name", "column_name", "interaction_weight", "interaction_value"]
DEFAULTS = [[0], [0], [0.0], ["null"], ["null"], [0.0], [0.0]]
LABEL_COL = "interaction_value"


def get_feature_columns(vocab_size, embedding_size=64):
    row_fc = tf.feature_column.categorical_column_with_identity("row_id", vocab_size, 0)
    column_fc = tf.feature_column.categorical_column_with_identity("column_id", vocab_size, 0)

    row_embed = tf.feature_column.embedding_column(row_fc, embedding_size)
    column_embed = tf.feature_column.embedding_column(column_fc, embedding_size)
    shared_embed = tf.feature_column.shared_embedding_columns([row_fc, column_fc], embedding_size)

    row_bias = tf.feature_column.embedding_column(row_fc, 1)
    column_bias = tf.feature_column.embedding_column(column_fc, 1)
    shared_bias = tf.feature_column.shared_embedding_columns([row_fc, column_fc], 1)

    return {
        "row_embed": row_embed,
        "column_embed": column_embed,
        "shared_embed": shared_embed,
        "row_bias": row_bias,
        "column_bias": column_bias,
        "shared_bias": shared_bias,
    }


def get_input_fn(csv_path, mode=tf.estimator.ModeKeys.TRAIN, batch_size=32):
    def input_fn():
        def parse_csv(value):
            columns = tf.decode_csv(value, DEFAULTS)
            features = dict(zip(COLUMNS, columns))
            label = features.pop(LABEL_COL)
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


def serving_input_fn(string_features=("row_name", "column_name")):
    features = {
        key: tf.placeholder(tf.string, [None])
        for key in string_features
    }
    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=features
    )

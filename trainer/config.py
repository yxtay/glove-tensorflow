# configs
TRAIN_CSV = "data/interaction.csv"
VOCAB_TXT = "data/vocab.txt"
EMBEDDING_SIZE = 64
L2_REG = 0.01
OPTIMIZER = "Adamax"
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
TRAIN_STEPS = 16384
STEPS_PER_EPOCH = 1024
TOP_K = 20

# field_names
ROW_ID = "row_token"
COL_ID = "col_token"
TARGET = "glove_value"
WEIGHT = "glove_weight"
# input_fn_args
COLUMN_NAMES = ["row_token_id", "col_token_id", "count", "value",
                "row_token", "col_token", "glove_weight", "glove_value"]
COLUMN_DEFAULTS = [[0], [0], [0.0], [0.0], [""], [""], [0.0], [0.0]]
FEATURE_NAMES = [ROW_ID, COL_ID]
TARGET_NAMES = [TARGET]
WEIGHT_NAMES = [WEIGHT]
# serving_input_fn_args
SERVING_STRING_FEATURES = [ROW_ID, COL_ID]

CONFIG = {
    "field_names": {
        "row_id": ROW_ID,
        "col_id": COL_ID,
        "target": TARGET,
    },
    "dataset_args": {
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "weight_names": WEIGHT_NAMES,
    },
    "serving_input_fn_args": {
        "string_features": SERVING_STRING_FEATURES,
    }
}

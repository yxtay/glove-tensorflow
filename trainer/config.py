# field_names
ROW_ID = "row_token"
COL_ID = "col_token"
WEIGHT = "glove_weight"
VALUE = "glove_value"
# input_fn_args
COLUMN_NAMES = ["row_token_id", "col_token_id", "count", "value",
                "row_token", "col_token", "glove_weight", "glove_value"]
COLUMN_DEFAULTS = [[0], [0], [0.0], [0.0], [""], [""], [0.0], [0.0]]
FEATURE_NAMES = [ROW_ID, COL_ID]
TARGET_NAMES = [VALUE]
WEIGHT_NAMES = [WEIGHT]
# serving_input_fn_args
SERVING_STRING_FEATURES = [ROW_ID, COL_ID]

CONFIG = {
    "field_names": {
        "row_id": ROW_ID,
        "col_id": COL_ID,
        "weight": WEIGHT,
        "value": VALUE,
    },
    "dataset_args": {
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "weight_names": WEIGHT_NAMES,
    },
    "input_fn_args": {
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "weight_names": WEIGHT_NAMES,
    },
    "serving_input_fn_args": {
        "string_features": SERVING_STRING_FEATURES,
    }
}

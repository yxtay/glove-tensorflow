# field_names
ROW_ID = "row_token"
COLUMN_ID = "column_token"
WEIGHT = "glove_weight"
VALUE = "glove_value"
# input_fn_args
COLUMN_NAMES = ["row_token_id", "column_token_id", "count", "value",
                "row_token", "column_token", "glove_weight", "glove_value"]
COLUMN_DEFAULTS = [[0], [0], [0.0], [0.0], [""], [""], [0.0], [0.0]]
LABEL_COL = "glove_value"
# serving_input_fn_args
SERVING_STRING_FEATURES = ["row_token", "column_token"]

CONFIG = {
    "field_names": {
        "row_id": ROW_ID,
        "column_id": COLUMN_ID,
        "weight": WEIGHT,
        "value": VALUE,
    },
    "input_fn_args": {
        "col_names": COLUMN_NAMES,
        "col_defaults": COLUMN_DEFAULTS,
        "label_col": LABEL_COL,
    },
    "serving_input_fn_args": {
        "string_features": SERVING_STRING_FEATURES,
    }
}

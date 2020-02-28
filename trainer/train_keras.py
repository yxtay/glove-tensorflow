import tensorflow as tf

from trainer.config import VOCAB_TXT, parse_args
from trainer.data_utils import get_csv_dataset, get_keras_estimator_input_fn, get_serving_input_fn
from trainer.model_utils import MatrixFactorisation, get_string_id_table
from trainer.train_utils import (
    get_eval_spec, get_exporter, get_keras_callbacks, get_keras_estimator, get_optimizer, get_train_spec,
)
from trainer.utils import file_lines


def build_compile_model(params):
    # init layers
    mf_layer = MatrixFactorisation(
        file_lines(params["vocab_txt"]),
        params["embedding_size"],
        params["l2_reg"],
        name="glove_value"
    )

    # build model
    inputs = [tf.keras.Input((), name=name) for name in [params["row_name"], params["col_name"]]]
    glove_value = mf_layer(inputs)
    glove_model = tf.keras.Model(inputs, glove_value, name="glove_model")

    # compile model
    optimizer = get_optimizer(params["optimizer"], learning_rate=params["learning_rate"])
    glove_model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    return glove_model


def get_glove_dataset(row_col_names, vocab_txt=VOCAB_TXT, **kwargs):
    string_id_table = get_string_id_table(vocab_txt)

    def lookup(features, targets, weights):
        features = {name: string_id_table.lookup(features[name], name=name + "_lookup")
                    for name in row_col_names}
        return features, targets, weights

    dataset = get_csv_dataset(**kwargs).map(lookup, num_parallel_calls=-1)
    return dataset


def main():
    params = parse_args()

    # build & compile model
    model = build_compile_model(params)
    estimator = get_keras_estimator(model, params["job_dir"])

    # keras
    # datasets
    train_dataset = get_glove_dataset(**params["dataset_args"], num_epochs=None)
    validation_dataset = get_glove_dataset(**params["dataset_args"])

    # train and evaluate
    model.fit(
        train_dataset,
        epochs=params["train_steps"] // params["steps_per_epoch"],
        callbacks=get_keras_callbacks(params["job_dir"]),
        validation_data=validation_dataset,
        steps_per_epoch=params["steps_per_epoch"],
    )

    # keras estimator
    # input functions
    dataset_args = {"dataset_fn": get_glove_dataset, **params["dataset_args"]}
    train_input_fn = get_keras_estimator_input_fn(**dataset_args, num_epochs=None)
    eval_input_fn = get_keras_estimator_input_fn(**dataset_args)

    # train, eval spec
    train_spec = get_train_spec(train_input_fn, params["train_steps"])
    exporter = get_exporter(get_serving_input_fn(**params["serving_input_fn_args"]))
    eval_spec = get_eval_spec(eval_input_fn, exporter)

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

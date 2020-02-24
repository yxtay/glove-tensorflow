import tensorflow as tf

from trainer.config import parse_args
from trainer.model_utils import MatrixFactorisation, get_glove_dataset
from trainer.train_utils import get_keras_callbacks, get_loss_fn, get_optimizer
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
    glove_model.compile(optimizer=optimizer, loss=get_loss_fn("MeanSquaredError"))
    return glove_model


def main():
    params = parse_args()

    # build & compile model
    model = build_compile_model(params)

    # set up train, validation dataset
    train_dataset = get_glove_dataset(**params["dataset_args"], num_epochs=None)
    validation_dataset = get_glove_dataset(**params["dataset_args"])

    # train and evaluate
    history = model.fit(
        train_dataset,
        epochs=params["train_steps"] // params["steps_per_epoch"],
        callbacks=get_keras_callbacks(params["job_dir"]),
        validation_data=validation_dataset,
        steps_per_epoch=params["steps_per_epoch"],
    )

    # # estimator
    # estimator = get_keras_estimator(model, params["job_dir"])
    #
    # # input functions
    # dataset_args = {"dataset_fn": get_glove_dataset, **dataset_args}
    # train_input_fn = get_keras_estimator_input_fn(**dataset_args, num_epochs=None)
    # eval_input_fn = get_keras_estimator_input_fn(**dataset_args)
    #
    # # train, eval spec
    # train_spec = get_train_spec(train_input_fn, params["train_steps"])
    # exporter = get_exporter(get_serving_input_fn(**CONFIG["serving_input_fn_args"]))
    # eval_spec = get_eval_spec(eval_input_fn, exporter)
    #
    # # train and evaluate
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

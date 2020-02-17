from trainer.glove_utils import build_glove_model, get_glove_dataset, init_params, parse_args
from trainer.utils import get_keras_callbacks, get_loss_fn, get_optimizer


def main():
    args = parse_args()
    params = init_params(args.__dict__)

    # set up model and compile
    model = build_glove_model(params["vocab_txt"], params["embedding_size"], params["l2_reg"])
    model.compile(optimizer=get_optimizer(params["optimizer"], learning_rate=params["learning_rate"]),
                  loss=get_loss_fn("MeanSquaredError"))

    # set up train, validation dataset
    dataset_args = {
        "file_pattern": params["train_csv"],
        "vocab_txt": params["vocab_txt"],
        "batch_size": params["batch_size"],
    }
    train_dataset = get_glove_dataset(**dataset_args, num_epochs=None)
    validation_dataset = get_glove_dataset(**dataset_args)

    # train and evaluate
    history = model.fit(
        train_dataset,
        epochs=params["train_steps"] // params["steps_per_epoch"],
        callbacks=get_keras_callbacks(params["job_dir"]),
        validation_data=validation_dataset,
        steps_per_epoch=params["steps_per_epoch"],
    )


if __name__ == '__main__':
    main()

from typing import Tuple
import tensorflow as tf
import numpy as np
from data import Dataset
from config import Config
from preprocessor import Preprocessor
import keras

from model_utils import load_data, initialize_model
from helpers import generate_minibatches, average_distance_error
from rits import RITS


def step(
    model: RITS,
    optimizer: keras.optimizers.optimizer_v2.optimizer_v2.OptimizerV2,
    training_set_minibatch: Dataset,
) -> np.ndarray:
    """A single iteration during training.

    Args:
        model (RITS): The initialized unidirectional RITS model.
        optimizer (Any): The training optimizer.
        training_set_minibatch (Dataset): The training set of trajectories, the mask and delta vector.

    Returns:
        np.ndarray: An array of the loss values of all samples in the training set.
    """
    with tf.GradientTape() as tape:
        _, custom_loss = model(
            training_set_minibatch.trajectories,
            training_set_minibatch.masks,
            training_set_minibatch.deltas,
        )
    gradients = tape.gradient(custom_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return custom_loss


def compute_errors(model: RITS, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the average model error on the specified dataset.

    Args:
        model (RITS): The trained RITS model.
        dataset (Dataset): The dataset used for evaluating the model.

    Returns:
        np.ndarray: The model loss on the dataset.
        np.ndarray: The masked error i.e., only on the missing/downsampled values.
    """
    predicted_imputations, loss = model(
        dataset.trajectories, dataset.masks, dataset.deltas
    )
    predicted_imputations = predicted_imputations.numpy()
    masked_error = average_distance_error(
        predicted_imputations, dataset.trajectories, dataset.masks
    )
    return np.mean(loss), masked_error


def train(config: Config) -> None:
    """Trains the model with the specified parameters.

    Args:
        config (Config): A collection of model parameters.
    """
    subtrips, masks = load_data("data")
    P = Preprocessor(config)
    training_set = P.preprocess(subtrips, masks, training=True)
    model, optimizer = initialize_model(config, training_set)

    for i in range(1, config.epochs + 1):
        for training_set_minibatch in generate_minibatches(
            training_set, config.batch_size, shuffle=True
        ):
            step(model, optimizer, training_set_minibatch)

        train_loss, train_error = compute_errors(model, training_set)

        print(
            f"Epoch {i}/{config.epochs}, Train Loss: {train_loss}, Train Error: {train_error}"
        )

        if i % config.checkpoint_freq == 0:
            model.save_weights(f"models/{config.model_name}/model")
            print("-- Model Saved --")
        print("-" * 10)


if __name__ == "__main__":
    train(Config())

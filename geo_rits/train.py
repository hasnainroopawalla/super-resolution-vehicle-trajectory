from typing import Dict, Any
import tensorflow as tf
import numpy as np
from pathlib import Path
from preprocessor import Preprocessor
from model_utils import save_model

from model_utils import load_data, initialize_model
from helpers import generate_minibatches, average_distance_error
from rits import RITS
from history import History


def step(
    model: RITS, optimizer: Any, X: np.ndarray, masks: np.ndarray, deltas: np.ndarray
) -> np.ndarray:
    """A single iteration during training.

    Args:
        model (RITS): The initialized unidirectional or bidirectional model.
        optimizer (Any): The training optimizer.
        X (np.ndarray): Training data.
        masks (np.ndarray): Training masks.
        deltas (np.ndarray): Training deltas.

    Returns:
        np.ndarray: An array of the loss values of all samples in the training data.
    """
    with tf.GradientTape() as tape:
        _, custom_loss = model(X, masks, deltas)
    gradients = tape.gradient(custom_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return custom_loss


def compute_errors(model, X, masks, deltas):
    predicted_imputations, loss = model(X, masks, deltas)
    predicted_imputations = predicted_imputations.numpy()
    error = average_distance_error(predicted_imputations, X, masks)
    return np.mean(loss), error[0], error[1]


def train(params: Dict[str, Any]) -> None:
    """Trains the model with the given parameters.

    Args:
        params (Dict[str, Any]): A collection of model parameters.
    """
    subtrips, masks = load_data("data")

    P = Preprocessor(params)
    train = P.preprocess(subtrips, masks, training=True)

    model, optimizer = initialize_model(params, train)

    model_dir = f'models/{params["model_name"]}'
    history = History(model_dir)

    for i in range(1, params['epochs'] + 1):
        for X_mb, masks_mb, deltas_mb in generate_minibatches(train.trajectories, train.masks, train.deltas, params['batch_size'], shuffle=True):
            step(model, optimizer, X_mb, masks_mb, deltas_mb)

        train_loss, train_entire_trajectory_error, train_masked_trajectory_error = compute_errors(model, train.trajectories, train.masks, train.deltas)
        history.update(train_loss, train_entire_trajectory_error, train_masked_trajectory_error)
        
        print(f'Epoch {i}/{params["epochs"]}, Train Loss: {train_loss}')
        print(f'Complete Error => Train: {train_entire_trajectory_error}')
        print(f'Masked Error => Train: {train_masked_trajectory_error}')

        if i % params['checkpoint_freq'] == 0:
            save_model(model_dir, model, history)

        print('-'*10)


def get_params():
    params = {
        "model_name": "model",
        "traj_len": 30,
        "stride": 4,
        "traj_dist": 1000, # meters
        "epochs": 500,
        "learning_rate": 1e-4,
        "batch_size": 128,
        "hid_dim": 100,
        "load_weights": False,
        "unidirectional": True,
        "checkpoint_freq": 10,
        "normalize": True,
        "training": True,
    }
    model_dir = f"models/{params['model_name']}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return params


if __name__ == "__main__":
    train(get_params())

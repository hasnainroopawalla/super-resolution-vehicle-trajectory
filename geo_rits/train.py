from typing import Dict, Any
import tensorflow as tf
import numpy as np
from pathlib import Path
from model_utils import save_model

from model_utils import preprocess, load_data, initialize_model
from helpers import generate_minibatches, average_distance_error
from rits import RITS
from history import History


def step(model: RITS, optimizer: Any, X: np.ndarray, masks: np.ndarray, deltas: np.ndarray) -> np.ndarray:
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
    train_subtrips, train_subtrips_downsampled, test_subtrips, test_subtrips_downsampled = load_data('data')
    X_train, train_masks, train_deltas, X_val, val_masks, val_deltas, _, _, _ = preprocess(params, train_subtrips, train_subtrips_downsampled, test_subtrips, test_subtrips_downsampled)
    print('Train', X_train.shape)
    print('Val', X_val.shape)
    
    model, optimizer = initialize_model(params, X_train, train_masks, train_deltas)
    
    model_dir = f'models/{params["model_name"]}'
    history = History(model_dir)

    for i in range(1, params['epochs'] + 1):
        for X_mb, masks_mb, deltas_mb in generate_minibatches(X_train, train_masks, train_deltas, params['batch_size'], shuffle=True):
            step(model, optimizer, X_mb, masks_mb, deltas_mb)

        train_loss, train_entire_trajectory_error, train_masked_trajectory_error = compute_errors(model, X_train, train_masks, train_deltas)
        val_loss, val_entire_trajectory_error, val_masked_trajectory_error = compute_errors(model, X_val, val_masks, val_deltas)
        history.update(train_loss, train_entire_trajectory_error, train_masked_trajectory_error, val_loss, val_entire_trajectory_error, val_masked_trajectory_error)

        print(f'Epoch {i}/{params["epochs"]}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        print(f'Complete Error => Train: {train_entire_trajectory_error}, Val Error: {val_entire_trajectory_error}')
        print(f'Masked Error => Train: {train_masked_trajectory_error}, Val Error: {val_masked_trajectory_error}')

        if i % params['checkpoint_freq'] == 0:
            save_model(model_dir, model, history)

        print('-'*10)

def get_params():
    params = {
        'traj_len': 30,
        'stride': 2,
        'traj_dist': 1000, # meters
        'epochs': 500,
        'learning_rate': 1e-4,
        'batch_size': 128,
        'unidirectional': True,
        'hid_dim': 100,
        'load_weights': True,
        'checkpoint_freq': 10,
        'normalize': True,
        'training': True
    }
    params['model_name'] = '_'.join(str(val) for key, val in list(params.items()) if key in ['traj_len', 'stride', 'traj_dist', 'learning_rate', 'batch_size', 'unidirectional', 'hid_dim', 'normalize'])
    model_dir = f"models/{params['model_name']}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return params

if __name__ == '__main__':
    train(get_params())
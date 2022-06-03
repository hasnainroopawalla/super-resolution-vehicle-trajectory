from typing import Any, Dict, Tuple, Union
import numpy as np
import tqdm
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf

from rits import RITS
from brits import BRITS
from helpers import get_short_distance_trajectories_indices, get_start_end_missing_trajectories_indices, filter_trajectories, minmax_normalize


def create_trajectories(subtrips: pd.DataFrame, trajectory_len: int, stride: int) -> np.ndarray:
    """Creates shorter sequences/trajectories of the subtrips using the sliding window approach.

    Args:
        subtrips (pd.DataFrame): A collection of pre-processed subtrips.
        trajectory_len (int): The number of samples in each trajectory (sliding window size).
        stride (int): The number of steps the sliding window moves by each iteration. 

    Returns:
        np.ndarray: An array of all the generated trajectories from the subtrips.
    """
    all_trajectories = np.array([])
    for subtrip in tqdm.tqdm(subtrips):
        if len(subtrip) < trajectory_len: 
            continue
        subtrip_subset = subtrip[['latitude', 'longitude']].to_numpy().astype('float32')
        trajectories = np.squeeze(sliding_window_view(subtrip_subset, (trajectory_len, 2))[::stride], axis=1)
        if len(all_trajectories) == 0:
            all_trajectories = trajectories
        else:
            all_trajectories = np.vstack((all_trajectories, trajectories))
    return all_trajectories


def create_mask_vector(downsampled_trajectories: np.ndarray) -> np.ndarray:
    """Creates the masks vector 'm' given the downsampled trajectories. 
    The vector contains 1 for observed values and 0 for missing values.

    Args:
        downsampled_trajectories (np.ndarray): An array of all the generated trajectories from the subtrips.

    Returns:
        np.ndarray: The mask vector 'm'.
    """
    return (~np.isnan(downsampled_trajectories) * 1).astype('float32') 


def create_delta_vector(masks: np.ndarray) -> np.ndarray:
    """Creates the delta vector given the masks.  
    This vector represents the time gaps in the input series between observed values.

    Args:
        masks (np.ndarray): The mask vector 'm'.

    Returns:
        np.ndarray: The delta vector.
    """
    deltas = []
    for mask in tqdm.tqdm(masks):
        delta = []
        for i in range(len(mask)):
            if i == 0:
                delta.append(0)
            else:
                if mask[i-1][0] == 0:
                    delta.append(1 + delta[-1])
                elif mask[i-1][0] == 1:
                    delta.append(1)
        delta = np.array([delta])
        delta = np.repeat(delta, 2, axis=0)
        deltas.append(delta)
    deltas = np.array(deltas).astype('float32')
    deltas = np.swapaxes(deltas, 1, 2)
    return deltas
    

def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the data (subtrips) from the specified file path.

    Args:
        data_path (str): The path to the data folder.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training Data, Downsampled Training Data, Test Data, Downsampled Test Data.
    """
    print('Loading Data..')
    with open(f'{data_path}/train_subtrips_clean.data', 'rb') as filehandle:
        train_subtrips_stockholm = pickle.load(filehandle)

    with open(f'{data_path}/train_subtrips_downsampled_clean.data', 'rb') as filehandle:
        train_subtrips_stockholm_downsampled = pickle.load(filehandle)

    with open(f'{data_path}/test_subtrips_clean.data', 'rb') as filehandle:
        test_subtrips_stockholm = pickle.load(filehandle)

    with open(f'{data_path}/test_subtrips_downsampled_clean.data', 'rb') as filehandle:
        test_subtrips_stockholm_downsampled = pickle.load(filehandle)

    return train_subtrips_stockholm, train_subtrips_stockholm_downsampled, test_subtrips_stockholm, test_subtrips_stockholm_downsampled


def load_normalizing_params(model_path: str) -> Dict[str, float]:
    """Loads the parameters used for normalization.

    Args:
        model_path (str): The path to the model folder.

    Returns:
        Dict[str, float]: The normalizing parameters required for denormalization.
    """
    with open(f'{model_path}/normalizing_params.data', 'rb') as filehandle:
        normalizing_params = pickle.load(filehandle)

    return normalizing_params


def initialize_model(params: Dict[str, Any], X_train: np.ndarray, train_masks: np.ndarray, train_deltas: np.ndarray) -> Tuple[Union[RITS, BRITS], Any]:
    """Initializes the model with the specified parameters.

    Args:
        params (Dict[str, Any]): A collection of model parameters.
        X_train (np.ndarray): Training data.
        train_masks (np.ndarray): Training masks.
        train_deltas (np.ndarray): Training deltas.

    Returns:
        Tuple[Union[RITS, BRITS]]: The initialized unidirectional or bidirectional model.
        Optimizer: The optimizer initialized for training.
    """
    if params['unidirectional']:
        model = RITS(internal_dim=2, hid_dim=params['hid_dim'])
    else:
        model = BRITS(internal_dim=2, hid_dim=params['hid_dim'])
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model(X_train[0:1], train_masks[0:1], train_deltas[0:1])

    if params['load_weights']:
        model.load_weights(f'models/{params["model_name"]}/model')
        print('Existing Weights Loaded.')

    print(model.summary())
    return model, optimizer


def reverse_trajectories(X: np.ndarray):
    return X[:, ::-1, :]


def save_model(model_dir, model, history):
    model.save_weights(f'{model_dir}/model')
    pickle.dump(history, open(f"{model_dir}/history.data", "wb"))
    print('-- Model Saved --')


def preprocess(params: Dict[str, Any], train_subtrips: np.ndarray, train_subtrips_downsampled: np.ndarray, test_subtrips: np.ndarray, test_subtrips_downsampled: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        params (Dict[str, Any]): A collection of model parameters.
        train_subtrips (np.ndarray): Training subtrips.
        train_subtrips_downsampled (np.ndarray): Downsampled training subtrips.
        test_subtrips (np.ndarray): Test subtrips.
        test_subtrips_downsampled (np.ndarray): Downsampled test subtrips.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The pre-processed trajectories.
    """
    print('Pre-processing..')
    # Create train trajectories of defined length
    X_train = create_trajectories(train_subtrips, params['traj_len'], params['stride'])
    X_train_downsampled = create_trajectories(train_subtrips_downsampled, params['traj_len'], params['stride'])

    # Reverse the trajectories and append them to the existing dataset
    X_train = np.concatenate((X_train, reverse_trajectories(X_train)))
    X_train_downsampled = np.concatenate((X_train_downsampled, reverse_trajectories(X_train_downsampled)))

    # Remove all trajectories where the truck barely travels
    indices = get_short_distance_trajectories_indices(X_train, params['traj_dist'])
    X_train = filter_trajectories(X_train, indices)
    X_train_downsampled = filter_trajectories(X_train_downsampled, indices)

    # Create a validation set
    X_train, X_val, X_train_downsampled, X_val_downsampled = train_test_split(X_train, X_train_downsampled, test_size=0.05, shuffle=True)

    # Create train mask and delta vectors
    train_masks = create_mask_vector(X_train_downsampled)
    train_deltas = create_delta_vector(train_masks)

    # Create validation mask and delta vectors
    val_masks = create_mask_vector(X_val_downsampled)
    val_deltas = create_delta_vector(val_masks)

    # Create test trajectories of defined length
    X_test = create_trajectories(test_subtrips, params['traj_len'], params['stride'])
    X_test_downsampled = create_trajectories(test_subtrips_downsampled, params['traj_len'], params['stride'])

    indices = get_short_distance_trajectories_indices(X_test, params['traj_dist'])
    X_test = filter_trajectories(X_test, indices)
    X_test_downsampled = filter_trajectories(X_test_downsampled, indices)

    # Create mask and delta vectors
    test_masks = create_mask_vector(X_test_downsampled)
    test_deltas = create_delta_vector(test_masks)
    
    # Filter all trajectories that start and end with a missing value
    train_indices = get_start_end_missing_trajectories_indices(train_masks)
    X_train = filter_trajectories(X_train, train_indices)
    train_masks = filter_trajectories(train_masks, train_indices)
    train_deltas = filter_trajectories(train_deltas, train_indices)

    val_indices = get_start_end_missing_trajectories_indices(val_masks)
    X_val = filter_trajectories(X_val, val_indices)
    val_masks = filter_trajectories(val_masks, val_indices)
    val_deltas = filter_trajectories(val_deltas, val_indices)

    test_indices = get_start_end_missing_trajectories_indices(test_masks)
    X_test = filter_trajectories(X_test, test_indices)
    test_masks = filter_trajectories(test_masks, test_indices)
    test_deltas = filter_trajectories(test_deltas, test_indices)

    if params['normalize']:
        normalizing_params = {}
        X_train, normalizing_params['X_train_min_val'], normalizing_params['X_train_max_val'] = minmax_normalize(X_train)
        X_val, normalizing_params['X_val_min_val'], normalizing_params['X_val_max_val'] = minmax_normalize(X_val)
        X_test, normalizing_params['X_test_min_val'], normalizing_params['X_test_max_val'] = minmax_normalize(X_test)
        if params['training']: 
            # Save normalizing params if the model is in training mode
            pickle.dump(normalizing_params, open(f"models/{params['model_name']}/normalizing_params.data", "wb"))
    
    return X_train, train_masks, train_deltas, X_val, val_masks, val_deltas, X_test, test_masks, test_deltas
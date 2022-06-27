import haversine as hs
from haversine import Unit
import numpy as np
from typing import Generator, Tuple, List, Dict
import statistics as st
import pickle
import keras
import tensorflow as tf

from data import Dataset
from config import Config
from rits import RITS


def generate_minibatches(
    dataset: Dataset, batch_size: int, shuffle: bool = False
) -> Generator[Dataset, None, None]:
    """Generate batches of the data based on the batch_size.
    This method also handles cases where the number of samples is not divisible by the batch size.
    Example:
        X.shape -> (50, 7)
        batch_size -> 15
        generate_minibatches(X, batch_size) -> batches returned of shapes (for X) -> (15, 7), (15, 7), (15, 7), (5, 7)
    Args:
        dataset (Dataset): The input dataset.
        batch_size (int): An integer which determines the size of each batch (number of samples in each batch).
        shuffle (bool, optional): A flag which determines if the training set should be shuffled before batches are created. Defaults to False.
    Yields:
        Dataset: A minibatch Dataset object.
    """
    if shuffle:
        indices = np.arange(dataset.trajectories.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, dataset.trajectories.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, dataset.trajectories.shape[0])
        if shuffle:
            batch = indices[start_idx:end_idx]
        else:
            batch = slice(start_idx, end_idx)
        dataset_minibatch = Dataset()
        dataset_minibatch.trajectories, dataset_minibatch.masks, dataset_minibatch.deltas = (
            dataset.trajectories[batch],
            dataset.masks[batch],
            dataset.deltas[batch],
        )
        yield dataset_minibatch


def haversine_distance(
    latitude1: float, longitude1: float, latitude2: float, longitude2: float
) -> float:
    """Computes the haversine distance between two coordinates.

    Args:
        latitude1 (float): Latitude of coordinate 1.
        longitude1 (float): Longitude of coordinate 1.
        latitude2 (float): Latitude of coordinate 2.
        longitude2 (float): Longitude of coordinate 2.

    Returns:
        float: The haversine distance in meters.
    """
    return hs.haversine(
        (latitude1, longitude1), (latitude2, longitude2), unit=Unit.METERS
    )


def average_distance_error(
    predicted_trajectories: np.ndarray,
    actual_trajectories: np.ndarray,
    masks: np.ndarray,
) -> Tuple[float, float]:
    """Computes the average distance error between the estimated trajectory and the actual trajectory for missing/downsampled values.

    Args:
        predicted_trajectories (np.ndarray): An array of predicted/estimated trajectories.
        actual_trajectories (np.ndarray): An array of ground-truth trajectories.
        masks (np.ndarray): The mask vector representing the observed and missing values in the trajectories.

    Returns:
        float: The average distance error on the missing/downsampled values.
    """
    masked_error = []
    for predicted, actual, mask in zip(
        predicted_trajectories, actual_trajectories, masks
    ):
        for predicted_value, actual_value, mask_value in zip(predicted, actual, mask):
            error = haversine_distance(
                predicted_value[0], predicted_value[1], actual_value[0], actual_value[1]
            )
            if mask_value[0] == 0:
                masked_error.append(error)
    return st.mean(masked_error)


def create_mask_vector(downsampled_trajectories: np.ndarray) -> np.ndarray:
    """Creates the masks vector given the downsampled trajectories.
    The vector contains 1 for observed values and 0 for missing values.

    Args:
        downsampled_trajectories (np.ndarray): An array of all the generated trajectories from the subtrips.

    Returns:
        np.ndarray: The mask vector.
    """
    return (~np.isnan(downsampled_trajectories) * 1).astype("float32")


def load_data(data_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Loads the subtrips and masks from the specified file path.

    Args:
        data_path (str): The path to the data folder.

    Returns:
        List[np.ndarray]: The subtrips.
        List[np.ndarray]: The mask vector for the subtrips obtained after adaptive downsampling.
    """
    print("Loading Data..")
    with open(f"{data_path}/subtrips.data", "rb") as filehandle:
        subtrips = pickle.load(filehandle)
    with open(f"{data_path}/masks.data", "rb") as filehandle:
        masks = pickle.load(filehandle)
    print("Done")
    return subtrips, masks


def load_normalizing_params(model_path: str) -> Dict[str, float]:
    """Loads the parameters used for normalization.

    Args:
        model_path (str): The path to the model folder.

    Returns:
        Dict[str, float]: The normalizing parameters required for denormalization.
    """
    with open(f"{model_path}/normalizing_params.data", "rb") as filehandle:
        normalizing_params = pickle.load(filehandle)
    return normalizing_params


def initialize_model(
    config: Config, dataset: Dataset
) -> Tuple[RITS, keras.optimizers.optimizer_v2.optimizer_v2.OptimizerV2]:
    """Initializes the model with the specified parameters.

    Args:
        config (Config): A collection of model parameters.
        dataset (Dataset): The dataset containing trajectories, the mask and the delta vectors.

    Returns:
        RITS: The initialized unidirectional RITS model.
        keras.optimizers.optimizer_v2.optimizer_v2.OptimizerV2: The optimizer initialized for training.
    """
    model = RITS(internal_dim=2, hid_dim=config.hid_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model(dataset.trajectories[0:1], dataset.masks[0:1], dataset.deltas[0:1])

    if config.load_weights:
        model.load_weights(f"models/{config.model_name}/model")
        print("Existing Weights Loaded.")

    print(model.summary())
    return model, optimizer


def compute_model_error(model: RITS, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
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

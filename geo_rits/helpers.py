import haversine as hs
from haversine import Unit
import numpy as np
from typing import Generator, Tuple
import statistics as st

from data import Dataset


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

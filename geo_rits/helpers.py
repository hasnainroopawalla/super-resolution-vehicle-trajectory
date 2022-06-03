import haversine as hs
from haversine import Unit
import numpy as np
from typing import Generator, List, Tuple
import statistics as st


def filter_trajectories(trajectories: np.ndarray, indices: List[int]) -> np.ndarray:
    """Retains all trajectories at the specified indices and discards the rest.

    Args:
        trajectories (np.ndarray): A collection of trajectories.
        indices (List[int]): A List of indices of the trajectories that need to be kept.

    Returns:
        np.ndarray: The filtered trajectories.
    """
    return np.array([trajectories[i] for i in indices])
 

def get_short_distance_trajectories_indices(trajectories: np.ndarray, traj_dist: int) -> np.ndarray:
    """Removes all trajectories where the truck travels less than a threshold.

    Args:
        X (np.ndarray): The input trajectories.
        traj_dist (int, optional): The minimum distance threshold in meters.

    Returns:
        np.ndarray: An array only containing trajectories where the truck travels more than the distance threshold.
    """
    indices = []
    for idx, seq in enumerate(trajectories):
        dist_covered = 0
        for sample in range(1, len(seq)):
            dist_covered += hs.haversine((seq[sample][0], seq[sample][1]), (seq[sample-1][0], seq[sample-1][1]), unit=Unit.METERS)
        if dist_covered >= traj_dist:
            indices.append(idx)
    return indices


def get_start_end_missing_trajectories_indices(trajectories: np.ndarray) -> np.ndarray:
    """Discards all trajectories where the first and last sample is missing.

    Args:
        trajectories (np.ndarray): The input trajectories.

    Returns:
        np.ndarray: An array only containing trajectories where the first and last sample are observed.
    """
    indices = []
    for idx, seq in enumerate(trajectories):
       if seq[0][0] == 1:# and seq[-1][0] == 1:
            indices.append(idx)
    return indices


def minmax_normalize(X: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Performs Min-Max Normalization on the array.

    Args:
        X (np.ndarray): The input array.

    Returns:
        Tuple[np.ndarray, float, float]: The normalized array along with the min and max values required for denormalization.
    """
    min_val = X.min(axis=(0, 1), keepdims=True)
    max_val = X.max(axis=(0, 1), keepdims=True)
    return (X - min_val)/(max_val - min_val), min_val, max_val


def minmax_denormalize(z: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Performs Min-Max denormalization on the normalized array.

    Args:
        z (np.ndarray): The normalized array.
        min_val (float): The minimum value of the array.
        max_val (float): The maximum value of the array.

    Returns:
        np.ndarray: The denormalized array.
    """
    return z * (max_val - min_val) + min_val


def z_score_normalize(X: np.ndarray):
    """Performs Z-Score Normalization on the array.

    Args:
        X (np.ndarray): The input array.

    Returns:
        Tuple[np.ndarray, float, float]: The normalized array along with the mean and standard deviation values required for denormalization.
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    return (X - mean)/std, mean, std


def z_score_denormalize(z: np.ndarray, mean: float, std: float):
    """Performs Z-Score denormalization on the normalized array.

    Args:
        z (np.ndarray): The normalized array.
        mean (float): The mean of the array.
        std (float): The standard deviation of the array.

    Returns:
        np.ndarray: The denormalized array.
    """
    return z * std + mean


def generate_minibatches(
    X: np.ndarray, masks: np.ndarray, deltas: np.ndarray, batch_size: int, shuffle: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Generate batches of the data based on the batch_size.
    This method also handles cases where the number of samples is not divisible by the batch size.
    Example:
        X.shape -> (50, 7)
        y.shape -> (50, 1)
        batch_size -> 15
        generate_minibatches(X, y, batch_size) -> batches returned of shapes (for X) -> (15, 7), (15, 7), (15, 7), (5, 7)
    Args:
        X (np.ndarray): The input data.
        y (np.ndarray): The true labels of the data.
        batch_size (int): An integer which determines the size of each batch (number of samples in each batch).
        shuffle (bool, optional): A flag which determines if the training set should be shuffled before batches are created. Defaults to False.
    Yields:
        np.ndarray: Batches for X of size batch_size (same np.adarray format as X and y).
        np.ndarray: Batches for y of size batch_size (same np.adarray format as X and y).
    """
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        if shuffle:
            batch = indices[start_idx:end_idx]
        else:
            batch = slice(start_idx, end_idx)
        yield X[batch], masks[batch], deltas[batch]


def haversine_distance(latitude1: float, longitude1: float, latitude2: float, longitude2: float) -> float:
    """Computes the haversine distance between two coordinates.

    Args:
        latitude1 (float): Latitude of coordinate 1.
        longitude1 (float): Longitude of coordinate 1.
        latitude2 (float): Latitude of coordinate 2.
        longitude2 (float): Longitude of coordinate 2.

    Returns:
        float: The haversine distance in meters.
    """
    return hs.haversine((latitude1, longitude1), (latitude2, longitude2), unit=Unit.METERS)


def average_distance_error(predicted_trajectories: np.ndarray, actual_trajectories: np.ndarray, masks: np.ndarray) -> Tuple[float, float]:
    """Computes the average distance error between the estimated trajectory and the actual trajectory.

    Args:
        predicted_trajectories (np.ndarray): An array of predicted/estimated trajectories.
        actual_trajectories (np.ndarray): An array of ground-truth trajectories.
        masks (np.ndarray): An array representing the observed and missing values in the trajectories.

    Returns:
        Tuple[float, float, float]: The mean, standard deviation and median of the error values.
    """
    entire_trajectory_error, missing_values_error = [], []
    for predicted, actual, mask in zip(predicted_trajectories, actual_trajectories, masks):
        for predicted_value, actual_value, mask_value in zip(predicted, actual, mask):
            error = haversine_distance(predicted_value[0], predicted_value[1], actual_value[0], actual_value[1])
            if mask_value[0] == 0: 
                missing_values_error.append(error)
            entire_trajectory_error.append(error)
    return st.mean(entire_trajectory_error), st.mean(missing_values_error)

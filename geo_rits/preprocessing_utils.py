from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import haversine as hs
from haversine import Unit

from data import Dataset

# def downsample_subtrip(subtrip: pd.DataFrame):
#     # Adaptive Downsampling on a single subtrip.
#     subtrip_downsampled = subtrip.reset_index(drop=True)
#     i = 1
#     while i < subtrip_downsampled.shape[0] - 1:
#         if subtrip_downsampled.loc[i, 'triggertype'] == 3:
#             subtrip_downsampled.loc[i, 'latitude'] = None
#             subtrip_downsampled.loc[i, 'longitude'] = None
#             i = i + 2
#         else:
#             i = i + 1
#     return subtrip_downsampled


def create_trajectories_and_masks(dataset: Dataset, params: Dict) -> np.ndarray:
    """Creates shorter sequences/trajectories of the subtrips using the sliding window approach.

    Args:
        subtrips (pd.DataFrame): A collection of pre-processed subtrips.
        trajectory_len (int): The number of samples in each trajectory (sliding window size).
        stride (int): The number of steps the sliding window moves by each iteration.

    Returns:
        np.ndarray: An array of all the generated trajectories from the subtrips.
    """
    print("Creating trajectories and masks..")
    trajectories, masks = np.array([]), np.array([])
    for subtrip, mask in tqdm(
        zip(dataset.trajectories, dataset.masks), total=len(dataset.trajectories)
    ):
        if len(subtrip) < params["traj_len"]:
            continue
        trajectories_subset = np.squeeze(
            sliding_window_view(subtrip, (params["traj_len"], 2))[:: params["stride"]],
            axis=1,
        )
        masks_subset = np.squeeze(
            sliding_window_view(mask, (params["traj_len"], 2))[:: params["stride"]],
            axis=1,
        )
        if len(trajectories) == 0:
            trajectories = trajectories_subset
            masks = masks_subset
        else:
            trajectories = np.vstack((trajectories, trajectories_subset))
            masks = np.vstack((masks, masks_subset))
    dataset.trajectories, dataset.masks = trajectories, masks
    return dataset


def create_delta_vector(dataset: Dataset, params: Dict) -> np.ndarray:
    """Creates the delta vector given the masks.
    This vector represents the time gaps in the input series between observed values.

    Args:
        masks (np.ndarray): The mask vector 'm'.

    Returns:
        np.ndarray: The delta vector.
    """
    print("Creating deltas..")
    deltas = []
    for mask in tqdm(dataset.masks):
        delta = []
        for i in range(len(mask)):
            if i == 0:
                delta.append(0)
            else:
                if mask[i - 1][0] == 0:
                    delta.append(1 + delta[-1])
                elif mask[i - 1][0] == 1:
                    delta.append(1)
        delta = np.array([delta])
        delta = np.repeat(delta, 2, axis=0)
        deltas.append(delta)
    deltas = np.array(deltas).astype("float32")
    deltas = np.swapaxes(deltas, 1, 2)
    dataset.deltas = deltas
    return dataset


def reverse_trajectories(dataset: Dataset, params: Dict):
    print("Reversing and appending trajectories..")
    dataset.trajectories = np.concatenate(
        (dataset.trajectories, dataset.trajectories[:, ::-1, :])
    )
    dataset.masks = np.concatenate((dataset.masks, dataset.masks[:, ::-1, :]))
    dataset.deltas = np.concatenate((dataset.deltas, dataset.deltas[:, ::-1, :]))
    return dataset


def remove_short_distance_trajectories(dataset: Dataset, params: Dict) -> np.ndarray:
    """Removes all trajectories where the truck travels less than a threshold.

    Args:
        X (np.ndarray): The input trajectories.
        traj_dist (int, optional): The minimum distance threshold in meters.

    Returns:
        np.ndarray: An array only containing trajectories where the truck travels more than the distance threshold.
    """
    print("Removing short distance trajectories (traj_dist)..")
    indices = []
    for idx, seq in tqdm(
        enumerate(dataset.trajectories), total=len(dataset.trajectories)
    ):
        dist_covered = 0
        for sample in range(1, len(seq)):
            dist_covered += hs.haversine(
                (seq[sample][0], seq[sample][1]),
                (seq[sample - 1][0], seq[sample - 1][1]),
                unit=Unit.METERS,
            )
        if dist_covered >= params["traj_dist"]:
            indices.append(idx)
    dataset.trajectories = np.array([dataset.trajectories[i] for i in indices])
    dataset.masks = np.array([dataset.masks[i] for i in indices])
    dataset.deltas = np.array([dataset.deltas[i] for i in indices])
    return dataset


def remove_first_sample_missing_trajectories(
    dataset: Dataset, params: Dict
) -> np.ndarray:
    """Discards all trajectories where the first and last sample is missing.

    Args:
        trajectories (np.ndarray): The input trajectories.

    Returns:
        np.ndarray: An array only containing trajectories where the first and last sample are observed.
    """
    print("Removing first sample missing trajectories..")
    indices = []
    for idx, seq in tqdm(enumerate(dataset.masks), total=len(dataset.masks)):
        if seq[0][0] == 1 and seq[-1][0] == 1:
            indices.append(idx)
    dataset.trajectories = np.array([dataset.trajectories[i] for i in indices])
    dataset.masks = np.array([dataset.masks[i] for i in indices])
    dataset.deltas = np.array([dataset.deltas[i] for i in indices])
    return dataset


def minmax_normalize(dataset: Dataset, params: Dict) -> Tuple[np.ndarray, float, float]:
    """Performs Min-Max Normalization on the array.

    Args:
        X (np.ndarray): The input array.

    Returns:
        Tuple[np.ndarray, float, float]: The normalized array along with the min and max values required for denormalization.
    """
    min_val = dataset.trajectories.min(axis=(0, 1), keepdims=True)
    max_val = dataset.trajectories.max(axis=(0, 1), keepdims=True)
    dataset.trajectories = (dataset.trajectories - min_val) / (max_val - min_val)
    return dataset

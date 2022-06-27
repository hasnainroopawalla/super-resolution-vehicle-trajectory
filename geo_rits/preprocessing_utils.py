from typing import Dict
import pickle
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import haversine as hs
from haversine import Unit

from data import Dataset
from config import Config

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


def create_trajectories(dataset: Dataset, config: Config) -> Dataset:
    """Creates shorter sequences/trajectories of the subtrips using the sliding window approach.

    Args:
        dataset (Dataset): The dataset object.
        config (Config): The set of parameters used for training.

    Returns:
        Dataset: The dataset object after creating trajectories and the mask vector.
    """
    print("Creating trajectories and masks..")
    trajectories, masks = np.array([]), np.array([])
    for subtrip, mask in tqdm(
        zip(dataset.trajectories, dataset.masks), total=len(dataset.trajectories)
    ):
        if len(subtrip) < config.traj_len:
            continue
        trajectories_subset = np.squeeze(
            sliding_window_view(subtrip, (config.traj_len, 2))[:: config.stride], axis=1
        )
        masks_subset = np.squeeze(
            sliding_window_view(mask, (config.traj_len, 2))[:: config.stride], axis=1
        )
        if len(trajectories) == 0:
            trajectories = trajectories_subset
            masks = masks_subset
        else:
            trajectories = np.vstack((trajectories, trajectories_subset))
            masks = np.vstack((masks, masks_subset))
    dataset.trajectories, dataset.masks = trajectories, masks
    return dataset


def create_delta_vector(dataset: Dataset, config: Config) -> Dataset:
    """Creates the delta vector given the mask vector.
    This vector represents the time gaps in the input series between observed values.

    Args:
        dataset (Dataset): The dataset object.
        config (Config): The set of parameters used for training.

    Returns:
        Dataset: The dataset object with the computed delta vector.
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


def reverse_trajectories(dataset: Dataset, config: Config) -> Dataset:
    """Reverses the trajectories and appends them to original dataset.

    Args:
        dataset (Dataset): The dataset object.
        config (Config): The set of parameters used for training.

    Returns:
        Dataset: The dataset object after reversing and appending all trajectories.
    """
    print("Reversing and appending trajectories..")
    dataset.trajectories = np.concatenate(
        (dataset.trajectories, dataset.trajectories[:, ::-1, :])
    )
    dataset.masks = np.concatenate((dataset.masks, dataset.masks[:, ::-1, :]))
    dataset.deltas = np.concatenate((dataset.deltas, dataset.deltas[:, ::-1, :]))
    return dataset


def remove_short_distance_trajectories(dataset: Dataset, config: Config) -> Dataset:
    """Removes all trajectories where the truck travels less than a threshold (determined by traj_dist).

    Args:
        dataset (Dataset): The dataset object.
        config (Config): The set of parameters used for training.

    Returns:
        Dataset: The dataset object after discarding all trajectories of short distances.
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
        if dist_covered >= config.traj_dist:
            indices.append(idx)
    dataset.trajectories = np.array([dataset.trajectories[i] for i in indices])
    dataset.masks = np.array([dataset.masks[i] for i in indices])
    dataset.deltas = np.array([dataset.deltas[i] for i in indices])
    return dataset


def remove_first_sample_missing_trajectories(
    dataset: Dataset, config: Config
) -> Dataset:
    """Discards all trajectories where the first sample is missing.

    Args:
        dataset (Dataset): The dataset object.
        config (Config): The set of parameters used for training.

    Returns:
        Dataset: The dataset object after discarding trajectories where the first sample is missing.
    """
    print("Removing first sample missing trajectories..")
    indices = []
    for idx, seq in tqdm(enumerate(dataset.masks), total=len(dataset.masks)):
        if seq[0][0] == 1:
            indices.append(idx)
    dataset.trajectories = np.array([dataset.trajectories[i] for i in indices])
    dataset.masks = np.array([dataset.masks[i] for i in indices])
    dataset.deltas = np.array([dataset.deltas[i] for i in indices])
    return dataset


def minmax_normalize(dataset: Dataset, config: Config) -> Dataset:
    """Performs Min-Max Normalization on the dataset.

    Args:
        dataset (Dataset): The dataset object.
        config (Config): The set of parameters used for training.

    Returns:
        Dataset: The dataset object after normalizing the coordinate values and saving the min-max values required for denormalization.
    """
    print("Normalizing values..")
    min_val = dataset.trajectories.min(axis=(0, 1), keepdims=True)
    max_val = dataset.trajectories.max(axis=(0, 1), keepdims=True)
    dataset.trajectories = (dataset.trajectories - min_val) / (max_val - min_val)

    if config.training:
        # Save normalizing params if the model is in training mode
        pickle.dump(
            {"min_val": min_val, "max_val": max_val},
            open(f"models/{config.model_name}/normalizing_params.data", "wb"),
        )
    return dataset

from typing import Dict
import pandas as pd
import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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
    trajectories, masks = np.array([]), np.array([])
    for subtrip, mask in tqdm.tqdm(zip(dataset.data, dataset.masks), total=len(dataset.data)):
        if len(subtrip) < params['traj_len']: 
            continue
        trajectories_subset = np.squeeze(sliding_window_view(subtrip, (params['traj_len'], 2))[::params['stride']], axis=1)
        masks_subset = np.squeeze(sliding_window_view(mask, (params['traj_len'], 2))[::params['stride']], axis=1)
        if len(trajectories) == 0:
            trajectories = trajectories_subset
            masks = masks_subset
        else:
            trajectories = np.vstack((trajectories, trajectories_subset))
            masks = np.vstack((masks, masks_subset))
    dataset.data, dataset.masks = trajectories, masks
    return dataset


def create_delta_vector(dataset: Dataset, params: Dict) -> np.ndarray:
    """Creates the delta vector given the masks.  
    This vector represents the time gaps in the input series between observed values.

    Args:
        masks (np.ndarray): The mask vector 'm'.

    Returns:
        np.ndarray: The delta vector.
    """
    deltas = []
    for mask in tqdm.tqdm(dataset.masks):
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
    dataset.deltas = deltas
    return dataset
    
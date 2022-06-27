from typing import Callable, List
import numpy as np

from data import Dataset
from config import Config
from preprocessing_utils import (
    create_delta_vector,
    create_trajectories,
    reverse_trajectories,
    remove_short_distance_trajectories,
    remove_first_sample_missing_trajectories
    # minmax_normalize,
    ,
)


class Preprocessor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.training_jobs: List[Callable] = [
            create_trajectories,
            create_delta_vector,
            reverse_trajectories,
            remove_short_distance_trajectories,
            remove_first_sample_missing_trajectories
            # minmax_normalize,
        ]
        self.test_jobs: List[Callable] = []

    def preprocess(
        self, subtrips: List[np.ndarray], masks: List[np.ndarray], training: bool
    ) -> Dataset:
        """A method to preprocess the subtrips using the specified preprocessing pipeline.

        Args:
            subtrips (List[np.ndarray]): A list of subtrips.
            masks (List[np.ndarray]): The mask vector to indicate masked and observed values.
            training (bool): A flag to indicate if the preprocessing pipeline is applied to the training set or not.

        Returns:
            Dataset: An object of the dataset class containing the preprocessed trajectories, mask and delta vectors.
        """
        dataset = Dataset()
        dataset.trajectories, dataset.masks = subtrips, masks
        jobs = self.training_jobs if training else self.test_jobs
        for job in jobs:
            dataset = job(dataset, self.config)
            print(f"Data shape: {dataset.trajectories.shape}")
        return dataset

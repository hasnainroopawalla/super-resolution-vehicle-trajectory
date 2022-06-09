from dataclasses import dataclass
from typing import Dict
from data import Dataset
from preprocessing_utils import (
    create_delta_vector,
    create_trajectories_and_masks,
    reverse_trajectories,
    remove_short_distance_trajectories,
    remove_first_sample_missing_trajectories,
    minmax_normalize,
)


@dataclass
class Preprocessor:
    params: Dict
    training_jobs = [
        create_trajectories_and_masks,
        create_delta_vector,
        reverse_trajectories,
        remove_short_distance_trajectories,
        remove_first_sample_missing_trajectories,
        minmax_normalize,
    ]

    def preprocess(self, subtrips, masks):
        dataset = Dataset()
        dataset.trajectories, dataset.masks = subtrips, masks
        for i in self.training_jobs:
            dataset = i(dataset, self.params)
            print(dataset.trajectories.shape, dataset.masks.shape, dataset.deltas.shape)
        return dataset

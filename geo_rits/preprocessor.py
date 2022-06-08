from dataclasses import dataclass, field
from typing import Callable, Dict, List
from data import Dataset
from model_utils import create_mask_vector
from preprocessing_utils import create_delta_vector, create_trajectories_and_masks


@dataclass
class Preprocessor:
    params: Dict
    training_jobs = [
        create_trajectories_and_masks,
        create_delta_vector
    ]
    
    def preprocess(self, dataset: Dataset):
        for i in self.training_jobs:
            dataset = i(dataset, self.params)
        print(dataset.data.shape, dataset.masks.shape, dataset.deltas.shape)
        return dataset
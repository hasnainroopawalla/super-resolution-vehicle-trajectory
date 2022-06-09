import numpy as np


class Dataset:
    def __init__(self) -> None:
        self.trajectories = np.array([])
        self.masks = np.array([])
        self.deltas = np.array([])

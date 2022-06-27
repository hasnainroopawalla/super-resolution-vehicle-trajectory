import numpy as np


class Dataset:
    def __init__(self) -> None:
        self.trajectories: np.ndarray = np.array([])
        self.masks: np.ndarray = np.array([])
        self.deltas: np.ndarray = np.array([])

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Dataset:
    data: np.ndarray = field(default_factory=list)
    masks: np.ndarray = field(default_factory=list)
    deltas: np.ndarray = field(default_factory=list)
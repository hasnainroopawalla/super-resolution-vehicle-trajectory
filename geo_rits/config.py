from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "model"
    data_path: str = "data"
    traj_len: int = 30
    stride: int = 4
    traj_dist: int = 1000  # meters
    epochs: int = 500
    learning_rate: float = 1e-4
    batch_size: int = 128
    hid_dim: int = 100
    load_weights: bool = False
    checkpoint_freq: int = 10
    training: bool = True

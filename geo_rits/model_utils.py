from typing import Any, Dict, List, Tuple
import numpy as np
import pickle
import tensorflow as tf
from data import Dataset
import keras
from config import Config

from rits import RITS


def create_mask_vector(downsampled_trajectories: np.ndarray) -> np.ndarray:
    """Creates the masks vector 'm' given the downsampled trajectories.
    The vector contains 1 for observed values and 0 for missing values.

    Args:
        downsampled_trajectories (np.ndarray): An array of all the generated trajectories from the subtrips.

    Returns:
        np.ndarray: The mask vector.
    """
    return (~np.isnan(downsampled_trajectories) * 1).astype("float32")


def load_data(data_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Loads the subtrips and masks from the specified file path.

    Args:
        data_path (str): The path to the data folder.

    Returns:
        List[np.ndarray]: The subtrips.
        List[np.ndarray]: The mask vector for the subtrips obtained after adaptive downsampling.
    """
    print("Loading Data..")
    subtrips, masks = [], []

    with open(f"{data_path}/train_subtrips_clean.data", "rb") as filehandle:
        train_subtrips_stockholm = pickle.load(filehandle)

    for i in train_subtrips_stockholm:
        subtrips.append(i[["latitude", "longitude"]].to_numpy().astype("float32"))

    with open(f"{data_path}/train_subtrips_downsampled_clean.data", "rb") as filehandle:
        train_subtrips_stockholm_downsampled = pickle.load(filehandle)

    # Convert the downsampled subtrips to a mask vector
    for i in train_subtrips_stockholm_downsampled:
        masks.append(
            create_mask_vector(
                i[["latitude", "longitude"]].to_numpy().astype("float32")
            )
        )
    print("Done")
    return subtrips, masks


def load_normalizing_params(model_path: str) -> Dict[str, float]:
    """Loads the parameters used for normalization.

    Args:
        model_path (str): The path to the model folder.

    Returns:
        Dict[str, float]: The normalizing parameters required for denormalization.
    """
    with open(f"{model_path}/normalizing_params.data", "rb") as filehandle:
        normalizing_params = pickle.load(filehandle)
    return normalizing_params


def initialize_model(
    config: Config, dataset: Dataset
) -> Tuple[RITS, keras.optimizers.optimizer_v2.optimizer_v2.OptimizerV2]:
    """Initializes the model with the specified parameters.

    Args:
        config (Config): A collection of model parameters.
        dataset (Dataset): The dataset containing trajectories, the mask and the delta vectors.

    Returns:
        RITS: The initialized unidirectional RITS model.
        keras.optimizers.optimizer_v2.optimizer_v2.OptimizerV2: The optimizer initialized for training.
    """
    model = RITS(internal_dim=2, hid_dim=config.hid_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model(dataset.trajectories[0:1], dataset.masks[0:1], dataset.deltas[0:1])

    if config.load_weights:
        model.load_weights(f"models/{config.model_name}/model")
        print("Existing Weights Loaded.")

    print(model.summary())
    return model, optimizer

import importlib.resources

import numpy as np


def load_data(task):
    with (
        importlib.resources.files(__package__)
        .joinpath(f"{task}_data.npz")
        .open("rb") as f
    ):
        data = np.load(f)
    return data["factors"], data["emg"]

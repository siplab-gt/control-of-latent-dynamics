import numpy as np
from brian2 import Quantity
from jaxtyping import Float


def light_and_probe_coords(
    grid: Quantity, k: int
) -> tuple[Quantity, Float[np.ndarray, "k 3"], Quantity]:
    coords = grid
    groups = np.arange(len(coords)).reshape(k, -1)
    probe_coords = None
    return coords, groups, probe_coords

import numpy as np
from brian2 import Quantity
from jaxtyping import Float


def light_and_probe_coords(
    grid: Quantity, k: int, m: int
) -> tuple[Quantity, Float[np.ndarray, "k 3"], Quantity]:
    light_coords = grid
    light_groups = np.arange(len(light_coords)).reshape(k, -1)
    probe_coords = None
    return light_coords, light_groups, probe_coords

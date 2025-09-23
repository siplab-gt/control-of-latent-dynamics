"""Defines how light and probe coordinates are distributed.

Broken into a separate file so Snakemake can only rerun necessary changed code."""

import brian2.only as b2
import numpy as np
from brian2 import Quantity
from jaxtyping import Float


def light_and_probe_coords(
    grid: Quantity, k: int, m: int
) -> tuple[Quantity, Float[np.ndarray, "k 3"], Quantity]:
    light_coords = grid

    # Subdivide the grid into k square groups (k must be a perfect square)
    assert np.sqrt(k).is_integer(), "k must be a perfect square"
    n_side_grid = int(np.sqrt(len(light_coords)))
    n_side_group = int(np.sqrt(k))
    assert n_side_grid % n_side_group == 0, "k must divide total number of grid points"
    group_size = n_side_grid // n_side_group

    # Assign group indices in square blocks
    group_indices = np.arange(len(light_coords)).reshape(n_side_grid, n_side_grid)
    light_groups = []
    for i in range(n_side_group):
        for j in range(n_side_group):
            block = group_indices[
                i * group_size : (i + 1) * group_size,
                j * group_size : (j + 1) * group_size,
            ].reshape(-1)
            light_groups.append(block)
    light_groups = np.array(light_groups)

    # make evenly spaced grid over same space as light coords
    assert np.sqrt(m).is_integer(), "m must be a perfect square"
    n_side_grid = int(np.sqrt(len(grid)))
    assert n_side_grid % np.sqrt(m) == 0, "m must divide total number of grid points"
    n_side_probe = int(np.sqrt(m))
    grid_mm = grid / b2.mm
    probe_coords = (
        grid_mm.reshape(n_side_grid, n_side_grid, 3)[
            :: n_side_grid // n_side_probe, :: n_side_grid // n_side_probe, :
        ].reshape(-1, 3)
        * b2.mm
    )

    return light_coords, light_groups, probe_coords

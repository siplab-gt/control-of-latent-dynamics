"""Defines how light and probe coordinates are distributed.

Broken into a separate file so Snakemake can only rerun necessary changed code."""

import brian2.only as b2
import numpy as np
from brian2 import Quantity
from jaxtyping import Float


def light_and_probe_coords(
    grid: Quantity, k: int, m: int
) -> tuple[Quantity, list[list[int]], Quantity]:
    # only going to be light coords, since no probe on sufficient_stim
    n_side_grid = int(np.sqrt(len(grid)))
    i_grid = np.arange(len(grid)).reshape(n_side_grid, n_side_grid)

    # Keeping cutting groups in half to get k square/rectangular groups
    assert np.log2(k).is_integer(), "k must be a power of 2"
    n_cuts = int(np.log2(k))
    groups = [i_grid]
    for i_cut in range(n_cuts):
        axis = i_cut % 2  # alternate between x and y axis
        if axis == 0:
            # cut by rows (horizontally)
            mid = groups[0].shape[0] // 2
            new_groups = [g[0:mid, :] for g in groups] + [g[mid:, :] for g in groups]
        elif axis == 1:
            # cut by columns (vertically)
            mid = groups[0].shape[1] // 2
            new_groups = [g[:, 0:mid] for g in groups] + [g[:, mid:] for g in groups]
        groups = new_groups
    groups = [g.flatten() for g in groups]

    # no probe on sufficient_stim
    assert m == "NA" or m == 0
    # make evenly spaced grid over same space as light coords
    # assert np.sqrt(m).is_integer(), "m must be a perfect square"
    # n_side_grid = int(np.sqrt(len(grid)))
    # assert n_side_grid % np.sqrt(m) == 0, "m must divide total number of grid points"
    # n_side_probe = int(np.sqrt(m))
    # grid_mm = grid / b2.mm
    # probe_coords = (
    #     grid_mm.reshape(n_side_grid, n_side_grid, 3)[
    #         :: n_side_grid // n_side_probe, :: n_side_grid // n_side_probe, :
    #     ].reshape(-1, 3)
    #     * b2.mm
    # )

    return grid, groups, None

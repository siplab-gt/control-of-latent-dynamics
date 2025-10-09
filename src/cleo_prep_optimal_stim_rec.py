from itertools import combinations

import numpy as np
from brian2 import Quantity, mm
from jaxtyping import Float, UInt


def force_mix_partition(grid, k, m, rng):
    """returns i_stim, i_rec"""
    assert grid.shape == (k + m, 3)
    stim_mask = np.zeros((k + m,), dtype=bool)
    stim_mask[rng.choice(k + m, k, replace=False)] = True
    D2 = ((grid[:, None, :] - grid[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    Dinv = 1.0 / D2

    def energy(mask):
        same = mask[:, None] == mask[None, :]
        return (same * Dinv).sum() / 2  # divide by 2 for double count

    iters_since_improvement = 0
    while iters_since_improvement < 100:
        # Randomly select a pair of sites to swap
        s, r = (
            rng.choice(np.where(stim_mask)[0]),
            rng.choice(np.where(~stim_mask)[0]),
        )
        # Compute the energy difference
        # compute energy diff efficiently
        # changing channel s from stim to rec
        # E change is for new rec site - minus old stim site
        ds = ((stim_mask == 0) * Dinv[s] - (stim_mask == 1) * Dinv[s]).sum()
        # changing channel r from rec to stim
        dr = ((stim_mask == 1) * Dinv[r] - (stim_mask == 0) * Dinv[r]).sum()
        dE = ds + dr
        if dE < 0:
            # Accept the swap
            stim_mask[s], stim_mask[r] = stim_mask[r], stim_mask[s]
            iters_since_improvement = 0
        iters_since_improvement += 1
    return stim_mask, ~stim_mask


def light_and_probe_coords(
    grid: Quantity, k: int, m: int, rng
) -> tuple[Quantity, list[list[int]], Quantity]:
    i_light, i_probe = force_mix_partition(grid / mm, k, m, rng)
    light_coords = grid[i_light]
    light_groups = np.arange(len(light_coords)).reshape(k, -1).tolist()
    probe_coords = grid[i_probe]
    return light_coords, light_groups, probe_coords

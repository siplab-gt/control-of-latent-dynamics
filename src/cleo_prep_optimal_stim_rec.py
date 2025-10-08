from itertools import combinations

import numpy as np
from brian2 import Quantity, mm
from jaxtyping import Float, UInt


def distributed_partition(grid, k, m):
    """returns i_stim, i_rec"""
    assert grid.shape == (k + m, 3)
    # work with the (n, 3) grid shape, rather than (n, n, 3)
    # this should be flexible for non-square grids too.
    n_side = int(np.sqrt(len(grid)))
    stim_mask = np.indices((n_side, n_side)).sum(0) % 2  # initial checkerboard pattern
    # stim_mask = np.indices(grid.shape[:-1]).sum(0) % 2  # initial checkerboard pattern
    stim_mask = stim_mask.astype(bool).reshape((-1,))
    assert stim_mask.shape == (k + m,)
    n_stim_init = np.sum(stim_mask)
    stim_coords = np.argwhere(stim_mask)
    if n_stim_init > k:
        print(
            f"distributing {len(grid)} sites into {k} stim and {m} rec by flipping"
            f" {n_stim_init - k} stim sites to rec from initial {n_stim_init} stim sites"
        )
        # remove closest stim-stim pairs first
        dists = np.array(
            [np.linalg.norm(grid[a] - grid[b]) for a, b in combinations(stim_coords, 2)]
        )
        pairs = np.array(list(combinations(range(len(stim_coords)), 2)))
        order = np.argsort(dists)  # nearest first
        removed = set()
        for i, j in pairs[order]:
            # for idx in (i, j):
            #     if len(removed) < (n_stim_init - k) and idx not in removed:
            #         removed.add(idx)
            if len(removed) < (n_stim_init - k) and i not in removed:
                removed.add(i)
            if len(removed) >= (n_stim_init - k):
                break
        for idx in removed:
            stim_mask[stim_coords[idx]] = False
    elif n_stim_init < k:
        # add rec sites farthest from existing stim sites
        print(
            f"distributing {len(grid)} sites into {k} stim and {m} rec by flipping"
            f" {k - n_stim_init} rec sites to stim from initial {n_stim_init} stim sites"
        )
        rec_coords = np.argwhere(~stim_mask)
        dists = np.min(
            np.linalg.norm(
                grid[rec_coords][:, None, :] - grid[stim_coords][None, :, :], axis=-1
            ),
            axis=1,
        ).flatten()
        new_stim_coords = rec_coords[np.argsort(dists)[-(k - n_stim_init) :]]
        print(f"{new_stim_coords=}")
        for c in new_stim_coords:
            stim_mask[c] = True
    assert np.sum(stim_mask) == k
    assert np.sum(~stim_mask) == m
    return np.argwhere(stim_mask), np.argwhere(~stim_mask)


def light_and_probe_coords(
    grid: Quantity, k: int, m: int
) -> tuple[Quantity, list[list[int]], Quantity]:
    i_light, i_probe = distributed_partition(grid / mm, k, m)
    light_coords = grid[i_light]
    light_groups = list(np.arange(len(light_coords)).reshape(k, -1))
    probe_coords = grid[i_probe]
    return light_coords, light_groups, probe_coords

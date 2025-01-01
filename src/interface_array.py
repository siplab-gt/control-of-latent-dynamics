import brian2.only as b2
import numpy as np


def get_interface_array(array_side_length_mm, total_scale, array_n_per_side):
    grid_len = array_side_length_mm * total_scale
    # TODO: define grid size by beam spacing?
    x, y, z = np.meshgrid(
        np.linspace(-grid_len / 2, grid_len / 2, array_n_per_side),
        np.linspace(-grid_len / 2, grid_len / 2, array_n_per_side),
        0.6,  # TODO: what depth? just one layer? this is L4
        indexing="ij",
    )
    return np.stack((x, y, z), axis=-1).reshape(-1, 3) * b2.mm

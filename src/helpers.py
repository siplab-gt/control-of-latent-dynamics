import brian2.only as b2
import cleo
import cleo.utilities
import numpy as np


def seed_everything(seed: int, extra_seed_str=""):
    """Set seed for numpy, brian, and cleo

    Parameters
    ----------
    seed : int
        Seed to set
    extra_seed_str : str, optional
        String used to modify the base seed so that different steps in the pipeline
        don't use identical seeds, by default "".
    """
    extra_seed_int = sum(ord(c) for c in extra_seed_str)
    seed += extra_seed_int
    print(f"Seeding with {seed}")
    np.random.seed(seed)
    b2.seed(seed)
    cleo.utilities.set_seed(seed)
    return seed

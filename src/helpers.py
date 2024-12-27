import brian2.only as b2
import cleo
import cleo.utilities
import numpy as np


def seed_everything(seed):
    np.random.seed(seed)
    b2.seed(seed)
    cleo.utilities.set_seed(seed)

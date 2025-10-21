import importlib

import brian2.only as b2
import cleo
import cleo.utilities
import numpy as np
from jaxtyping import UInt


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
    print(f"Seeding with {seed} + {extra_seed_int} = {seed + extra_seed_int}")
    seed += extra_seed_int
    np.random.seed(seed)
    b2.seed(seed)
    cleo.utilities.set_seed(seed)
    return seed


def str2bool(value):
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Cannot convert {value!r} to bool")
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean string: {value!r}")


def setup_debugger(port=5678):
    import debugpy

    debugpy.listen(port)
    print(f"Waiting for debugger to attach on port {port}...")
    debugpy.wait_for_client()
    print("Debugger attached, continuing...")


class AttrDict(dict):
    """Dictionary subclass whose entries can be accessed like attributes."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __or__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        result = AttrDict(self)
        result.update(other)
        return result


def preprocess_params(params_dict) -> AttrDict:
    """Recursively convert all values in a params or wildcards dict to int or float if possible,
    and wrap dicts so keys are accessible as attributes.
    Also adds Brian units for params with unit suffixes"""
    new_dict = AttrDict()
    for k, v in params_dict.items():
        if isinstance(v, dict):
            new_v = preprocess_params(v)
        else:
            if isinstance(v, (int, float, np.integer, np.floating)):
                new_v = v
            else:
                try:
                    new_v = int(v)
                except (ValueError, TypeError):
                    try:
                        new_v = float(v)
                    except (ValueError, TypeError):
                        new_v = v
        if isinstance(v, (list, tuple, int, float, np.integer, np.floating)):
            # add brian units if applicable
            for suffix, unit in [
                ("_ms", b2.ms),
                ("_s", b2.second),
                ("_mV", b2.mvolt),
                ("_um", b2.um),
                ("_mm", b2.mm),
                ("_uW", b2.uwatt),
                ("_mWmm2", b2.mwatt / b2.mm2),
            ]:
                if k.endswith(suffix):
                    k = k[: -len(suffix)]
                    new_v *= unit
                    break
        new_dict[k] = new_v
    return new_dict


def load_factors_emg(npz_file):
    with open(npz_file, "rb") as f:
        npz = np.load(f)
        return npz.get("factors", None), npz["emg"]


def orth(A, tol=None):
    # Perform singular value decomposition
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    if tol is None:
        tol = max(A.shape) * np.finfo(s.dtype).eps * np.max(s)

    # Determine the rank based on the given tolerance
    rank = np.sum(s > tol)

    # Return the orthonormal basis
    Q = U[:, :rank]

    return Q

import gzip
import logging
import sys

import brian2.only as b2
import cleo
import cleo.utilities
import dill
import numpy as np
import snakemake
from jaxtyping import UInt

from src.log_redirector import setup_permanent_logging


_NORMALIZED_CLIP_DEFAULT = 1e6
"""Default ±sigma clip bound for `sanitize_normalized`. 1e6 sigmas is absurdly
permissive for real neural signals (which live within ~±10 sigma); anything
beyond it is upstream pathology (e.g. v3 plant fSNN `vinf` runaway at high-k
producing LFP values that overflow float64 post-normalization). The clip
keeps the downstream `.astype(np.float32)` from producing ±inf → NaN loss,
at the cost of that channel's contribution being meaningless."""


_DEGENERATE_STD_RATIO = 1e-3
"""Relative std below which `safe_std_compute` floors a channel to 1.0."""

_CONSTANT_TAIL_MAX_RATIO = 1e-2
"""Upper bound on a dropped channel's std relative to the kept channels' scale.

Deliberately looser than `_DEGENERATE_STD_RATIO`: this only has to separate "constant" from
"driven", and the two populations are three orders of magnitude apart (the legacy tail measures
~3e-4 of the kept scale, a driven channel ~1). The slack matters because the closed-loop NULL arm
has the weakest behavior amplitude in the pipeline, where a tighter bound leaves under 2x margin
and a weak-EMG cell would hard-fail on data that is in fact fine."""

N_BEHAVIOR_KEPT = 2
"""Leading behavior channels the reaching and cycling tasks actually drive.

Both tasks emit a 3rd behavior output pinned to the constant -0.5 (`src/fsnn/reaching.py`,
`src/fsnn/cycling.py`). Its std is ~1e-4, so `safe_std_compute` floors it to 1.0 and the merged
[LFP|behavior] readout then owes it a constant -mean/std ~ +0.5 through C.x. The output loss is
block-normalized (`usid/train/loss.py::_reduce_output`), giving the behavior block the same weight as
the whole LFP block: at nz=3 that channel takes 1/6 of the reconstruction objective to reproduce a
constant, and the LFP block falls from 91% of the objective to 50%. Every Z intake trims to these
channels.
"""


def drop_constant_behavior(Z: np.ndarray, name: str = "Z", log=None) -> np.ndarray:
    """Trim a behavior array to its informative leading channels.

    Idempotent: an array already at or below `N_BEHAVIOR_KEPT` channels is returned unchanged, so the
    helper is safe to apply at every Z intake without tracking which ones ran first.

    Args:
        Z: (T, n_behavior) behavior array, or a bare (n_behavior,) single-timestep measurement as the
            closed-loop observer reads it off the network each step.
        name: diagnostic label for the log line and the error message.
        log: optional logger; falls back to stderr.
    Returns:
        Z trimmed along its last axis, or Z unchanged when it has no channels to drop.
    Raises:
        ValueError: a dropped channel carries real variance, so the constant-tail assumption this
            trim rests on does not hold for these data.
    """
    Z = np.asarray(Z)
    if Z.shape[-1] <= N_BEHAVIOR_KEPT:
        return Z
    if Z.ndim == 1:
        return Z[:N_BEHAVIOR_KEPT]   # one timestep carries no variance to check the tail against
    if Z.ndim != 2:
        raise ValueError(f"{name}: expected a (T, n_behavior) or (n_behavior,) array, got shape {Z.shape}.")

    # float64 before the std: the tail sits at a constant ~-0.5, where float32 cancellation inflates
    # its std roughly 2x (7.4e-5 vs 3.2e-5 measured) and biases this check toward a false raise.
    s = np.std(Z.astype(np.float64), axis=0)
    kept_s = s[:N_BEHAVIOR_KEPT]
    usable = kept_s[np.isfinite(kept_s) & (kept_s > 0)]
    dropped = s[N_BEHAVIOR_KEPT:]
    if usable.size == 0:
        msg = (f"{name}: kept channels carry no usable variance (std={kept_s}); trimming without "
               f"verifying the dropped tail.")
        (log.warning if log is not None else lambda m: print(f"[WARN] {m}", file=sys.stderr))(msg)
        return Z[:, :N_BEHAVIOR_KEPT]
    kept_scale = float(np.median(usable))
    informative = np.isfinite(dropped) & (dropped > _CONSTANT_TAIL_MAX_RATIO * kept_scale)
    if informative.any():
        raise ValueError(
            f"{name}: channel(s) {np.flatnonzero(informative) + N_BEHAVIOR_KEPT} beyond the first "
            f"{N_BEHAVIOR_KEPT} carry variance (std={dropped[informative]}, kept-channel scale "
            f"{kept_scale:g}); they are not the constant legacy tail this trim assumes. The task's "
            f"behavior definition changed -- update N_BEHAVIOR_KEPT."
        )

    msg = (f"{name}: dropping {Z.shape[-1] - N_BEHAVIOR_KEPT} constant behavior channel(s) "
           f"(std={dropped}); keeping {N_BEHAVIOR_KEPT}.")
    if log is not None:
        log.info(msg)
    else:
        print(f"[INFO] {msg}", file=sys.stderr)
    return Z[:, :N_BEHAVIOR_KEPT]


def safe_std_compute(arr: np.ndarray, axis: int, name: str, log=None):
    """Compute mean + std with degenerate-channel fallback.

    Clamps any `std == 0` (constant channel), `std == inf/nan` (overflow),
    or non-finite `mean` to safe values (1.0 / 0.0 respectively). Warns when
    either triggers. Does NOT post-process the normalized array — see
    `sanitize_normalized` for that (separate concern).

    Args:
        arr: Input array (T, D) or similar.
        axis: Axis to reduce.
        name: Diagnostic label for warning messages.
        log: Optional logger. If None, prints to stderr.
    Returns:
        (mean, std) with guaranteed `std > 0` and finite mean.
    """
    m = np.mean(arr, axis=axis)
    s = np.std(arr, axis=axis)

    # A near-constant channel (std ~ 0 but not exactly 0) makes both (arr-mean)/std and the merged-readout
    # output bias d = -mean/std blow up — e.g. the reaching task's degenerate 3rd behavior dim has
    # std ~5e-5, giving d ~1e4, which poisons the z-scored controller tracking objective and the
    # full-output observer innovation. Treat any std that is a negligible fraction of the typical channel
    # std the same as an exactly-constant channel (clamped to 1.0 below).
    _finite_pos = s[np.isfinite(s) & (s > 0)]
    std_floor = _DEGENERATE_STD_RATIO * float(np.median(_finite_pos)) if _finite_pos.size else 0.0
    bad_std = ~np.isfinite(s) | (s <= std_floor)
    bad_mean = ~np.isfinite(m)
    if bad_std.any() or bad_mean.any():
        n_std = int(bad_std.sum())
        n_mean = int(bad_mean.sum())
        arr_max = (
            float(np.max(np.abs(arr[np.isfinite(arr)])))
            if np.any(np.isfinite(arr)) else float("nan")
        )
        msg = (
            f"{name}: {n_std} channel(s) with non-finite/zero std, "
            f"{n_mean} with non-finite mean (likely overflow from extreme "
            f"values; max |{name}|={arr_max:g}). Clamping std→1.0, mean→0.0 "
            f"on those channels so de-normalization stays finite. Check "
            f"upstream data for pathological magnitudes."
        )
        if log is not None:
            log.warning(msg)
        else:
            print(f"[WARN] {msg}", file=sys.stderr)
        s = np.where(bad_std, 1.0, s)
        m = np.where(bad_mean, 0.0, m)
    return m, s


def sanitize_normalized(
    arr_norm: np.ndarray,
    name: str,
    bound: float = _NORMALIZED_CLIP_DEFAULT,
    log=None,
) -> tuple[np.ndarray, int]:
    """Clip a normalized array to ±`bound` sigmas to survive float32 cast.

    Even after `safe_std_compute` clamps bad std to 1.0, the raw `(arr - mean)`
    term can carry astronomical magnitudes from upstream pathology (e.g., v3
    plant `vinf` runaway). Casting those to float32 produces ±inf → NaN loss.
    This helper replaces non-finite values with ±bound / 0 and clips the rest
    to [-bound, +bound], warning if anything was touched.

    Both sys_id-phase normalization (`model_fit.py`) and natural-electrode
    normalization (`src/model_fit/usid.py::_load_and_normalize_natural_data`)
    call this to keep the sanitization logic single-sourced.

    Returns:
        Tuple of (sanitized array, count of clipped/non-finite elements).
        `n_clipped > 0` is a pathology flag surfaced to summary.csv as
        `n_sanitize_clipped_*` so aggregation can exclude affected rows —
        a clipped channel's contribution is meaningless even though the
        array is finite.
    """
    overflow = ~np.isfinite(arr_norm) | (np.abs(arr_norm) > bound)
    n_clipped = int(overflow.sum())
    if n_clipped > 0:
        msg = (
            f"{name}: {n_clipped} normalized sample(s) exceed ±{bound:g} or "
            f"non-finite. Clipping in place so downstream float32 cast "
            f"doesn't overflow into NaN training loss. This typically flags "
            f"an unusable configuration (e.g., v3 plant × extreme k/m where "
            f"LFP blows up); the fit will proceed but the affected channel's "
            f"score is meaningless."
        )
        if log is not None:
            log.warning(msg)
        else:
            print(f"[WARN] {msg}", file=sys.stderr)
        arr_norm = np.nan_to_num(arr_norm, nan=0.0, posinf=bound, neginf=-bound)
        arr_norm = np.clip(arr_norm, -bound, bound)
    return arr_norm, n_clipped


class PassthroughIOProcessor(cleo.LatencyIOProcessor):
    """Minimal IOProcessor that provides sample_period but no control.

    Used for open-loop data collection where we need an IOProcessor
    (for LFP signal sample_period) but don't need actual control.
    """

    def process(self, state_dict: dict, t_samp: b2.Quantity) -> tuple[dict, b2.Quantity]:
        """Return empty control signals (no-op for open-loop data collection)."""
        return {}, 0 * b2.ms


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


def save_pkl_gz(obj, filename):
    assert filename.endswith(".pkl.gz")
    # compresslevel=1: these blobs are mostly incompressible float arrays (gzip ~1.07x),
    # so level 9 (the default) spends large CPU for ~0.5% size gain. Level 1 is ~4x faster
    # to write; load is level-agnostic (gzip auto-detects), so bytes round-trip identically.
    with gzip.open(filename, "wb", compresslevel=1) as f:
        dill.dump(obj, f)


def load_pkl_gz(filename):
    assert filename.endswith(".pkl.gz")
    with gzip.open(filename, "rb") as f:
        return dill.load(f)


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


def setup_logging(cfg, logfile):
    if logfile:
        log_fh = setup_permanent_logging(logfile)

    level = cfg["log_level"].upper()
    log_format = cfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARN)

    # Add new handler pointing to current sys.stderr (the Tee)
    handler = logging.StreamHandler(sys.stderr)
    if logfile:
        assert len(sys.stderr.files) == 2, "stderr should be Tee from log_redirector"
    handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(handler)

    # set lower level for my code
    logging.getLogger("src").setLevel(level)
    # needed for src.scripts called by snakemake
    logging.getLogger("__main__").setLevel(level)

    if logfile:
        return log_fh


def setup_debugger(debug=True, port=5678):
    if not str2bool(debug):
        return
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
        # recursively update chidren dicts as well
        for key, value in other.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = result[key] | value
            else:
                result[key] = value
        return result


def preprocess_params(params_dict) -> AttrDict:
    """Recursively convert all values in a params or wildcards dict to int or float if possible,
    and wrap dicts so keys are accessible as attributes.
    Also adds Brian units for params with unit suffixes"""

    # skip if already done
    if isinstance(params_dict, AttrDict):
        return params_dict

    # for a snakemake Namedlist, combine dictionaries as well as named params into just one dict
    if isinstance(params_dict, snakemake.io.Namedlist):
        consolidated = {}
        for k, v in params_dict._allitems():
            if k is None:
                assert isinstance(v, dict)
                consolidated |= v
            else:
                consolidated[k] = v
        params_dict = consolidated

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
        if isinstance(new_v, (list, tuple, int, float, np.integer, np.floating)):
            # add brian units if applicable
            for suffix, unit in [
                ("_ms", b2.ms),
                ("_s", b2.second),
                ("_mV", b2.mvolt),
                ("_um", b2.um),
                ("_mm", b2.mm),
                ("_uW", b2.uwatt),
                # ("_mWmm2", b2.mwatt / b2.mm2),  # since need to strip units constantly
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


def array_vals_by_t(arr: np.ndarray, t: b2.Quantity, dt: b2.Quantity) -> b2.Quantity:
    """Helper to index a brian TimedArray's values by time quantity t"""
    # TimedArray can do multiple timesteps, but not multiple variables
    # so we access from .values ourselves insted of ta(t)
    i = np.asarray(t / dt).astype(np.int32)
    i = np.clip(i, 0, arr.shape[0] - 1)
    return arr[i]
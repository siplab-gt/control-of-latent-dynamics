"""USID model fitting for the control-of-latent-dynamics pipeline.

Thin wrapper around the staged USID pipeline: model_fit.py owns standardization and hands this
function the already-normalized augmented outputs (YN=LFP, ZN=behavior) plus the RAW opto light and
the y/z/u stats. fit_lds_to_data assembles the concat NormStats and the pre-normalized DataBundle the
pipeline consumes, then delegates to scripts.cold_training.cold_full_run (natural -> unnatural ->
SubspaceLDS export). It returns the controllable SubspaceLDS the Controller consumes directly, plus the
reconstructed trajectories; model_fit.py stamps the normalization stats onto the returned object.

Usage in Snakefile:
    Set config["model_fit_method"] = "usid" to use this module.
"""
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

log = logging.getLogger(__name__)

# Add USID repo to path for imports
USID_REPO_PATH = os.environ.get("USID_REPO_ROOT")
if not USID_REPO_PATH:
    raise EnvironmentError(
        "USID_REPO_ROOT environment variable not set. "
        "Set it to the root of the siplab-gt/usid repository."
    )
USID_REPO_PATH = os.path.abspath(USID_REPO_PATH)
if USID_REPO_PATH not in sys.path:
    sys.path.insert(0, USID_REPO_PATH)

import torch

from scripts.cold_training import cold_full_run
from usid.data.sources.cold import load_cold_databundle
from usid.interface.data import NormStats
from usid.util.config import to_ns
from src.helpers import drop_constant_behavior, sanitize_normalized
from src.scripts.sys_id_expts import ExptData


def _cfg_to_plain(obj):
    """Recursively convert the loaded USID cfg (nested SimpleNamespace) into plain
    yaml-safe types, so model_fit.py can fold the resolved usid_hyperparams into the
    per-cell config_snapshot.yaml (drift-free config recording)."""
    if isinstance(obj, SimpleNamespace):
        return {k: _cfg_to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _cfg_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cfg_to_plain(v) for v in obj]
    return obj


def fit_lds_to_data(
    YN: np.ndarray,
    ZN: np.ndarray,
    UN: np.ndarray,
    XN: np.ndarray,
    n_x: int,
    cond_params: dict,
    base_params: dict,
    outfiles,
    config: dict,
):
    """Build a pre-normalized COLD DataBundle and delegate to cold_full_run.

    model_fit.py owns standardization: YN/ZN/UN/XN arrive already z-scored and the y/z/u stats live in
    `config`. This assembles the concat NormStats (the single source of the model output bias
    d = -(y_mean/y_std)) and the DataBundle, de-normalizes the opto light to raw mW/mm² so the frozen
    opsin sees physical units, then runs the staged pipeline via cold_full_run. model_fit.py stamps the
    normalization stats + linreg onto the returned SubspaceLDS and de-normalizes Zhat/Yhat with the same
    stats.

    Args:
        YN: (T, ny_lfp) normalized LFP of the opto (sys-id) recording.
        ZN: (T, nz) normalized behavior of the opto recording.
        UN: (T, k) normalized inputs (optogenetic; task inputs unsupported).
        XN: (T, n_fsnn) normalized fSNN latent factors (eval reference).
        n_x: target state dimension (the USID dims come from cfg, not this).
        cond_params: condition-specific parameters (seed, k).
        base_params: base experiment parameters.
        outfiles: output file paths (save_root derives from outfiles.lds_model).
        config: experiment config with y/z/u stats, use_opsin, opsin_checkpoint_path, natural_data_path.

    Returns:
        subspace_lds: SubspaceLDS (duck-types as LDS for the Controller).
        Zhat: (T, nz) predicted behavior (normalized gauge).
        Yhat: (T, ny_lfp) predicted LFP (normalized gauge).
        Xhat: (T, n_x) estimated latent states.
    """
    use_cpu = config.get("model_fit_device", "cpu").lower() == "cpu"
    device = torch.device("cpu" if use_cpu else "cuda")
    log.info(f"Using {device} for USID inline training")

    ny_lfp = YN.shape[1]
    nz = ZN.shape[1]
    k = UN.shape[1]
    k_opto = int(cond_params.get("k", k))

    # USID trains only on opto inputs (B1'/B2' map light -> state); task-input coupling is unimplemented.
    if k > k_opto:
        raise NotImplementedError(
            f"USID model_fit does not support model_task_inputs=True "
            f"(UN has {k} columns but k_opto={k_opto}). "
            f"Task input coupling is not implemented for USID."
        )

    seed = int(cond_params.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_opsin = bool(config['use_opsin'])

    # Observer fold wildcard (0 -> LFP-only observer, 1 -> full-output). preprocess_params casts the
    # wildcard string to int, so int() handles both an int and a str value.
    fold_behavior = bool(int(cond_params["fold_behavior"]))

    # Per-leg parallel multistart counts (model_fit wildcards). Each injects the matching top-level
    # multistart.*_restarts; >1 activates the successive-halving ensemble in that leg's stage.
    # preprocess_params casts the wildcard string to int.
    natural_restarts = int(cond_params["natural_restarts"])
    usid_restarts = int(cond_params["usid_restarts"])

    # Validation-loss regime (model_fit wildcard): selects the validation.loss_weights that drive best-val
    # checkpointing + multistart culling. delta_dom = pre-refactor criterion; rollout_dom = current default.
    val_regime = str(cond_params["val_regime"])

    # Opto-pool row length (model_fit wildcard): the length of the contiguous rows the opto recording is
    # cut into, which sets both the opsin GRU's free-run length and the windows each row yields.
    pool_row_len = int(cond_params["pool_row_len"])

    # === USID config: cold_usid.yaml + runtime dims + COLD usid_overrides. no_opsin is a per-setting
    # runtime kwarg (the BRAID-vs-USID sweep axis), so inject it here — train_unnatural reads it strictly.
    cfg = _load_usid_config(ny_lfp, k_opto, seed, config.get("usid_overrides"), nz=nz, fold_behavior=fold_behavior,
                            natural_restarts=natural_restarts, usid_restarts=usid_restarts, val_regime=val_regime,
                            pool_row_len=pool_row_len)
    cfg.no_opsin = not use_opsin

    # === Natural electrode data, normalized in the controller's Y/Z coordinates (model_fit stats) ===
    natural_data_path = config.get('natural_data_path')
    if not natural_data_path:
        raise ValueError(
            "natural_data_path not found in config. "
            "Ensure model_fit rule provides natural_data input when method=usid."
        )
    y_mean = config['y_mean']
    y_std = config['y_std']
    u_mean = config['u_mean']
    u_std = config['u_std']
    # z_mean/z_std normalize the natural behavior (natural_expt.Z) into the controller's Z coordinates —
    # the behavior block of the augmented [LFP|behavior] output the merged readout Cy predicts.
    z_mean = config.get('z_mean')
    z_std = config.get('z_std')
    if z_mean is None or z_std is None:
        raise ValueError("USID model_fit requires z_mean/z_std in config (thread them from "
                         "model_fit.py) so the natural behavior loads in Z coords for the merged Cy.")
    natural_YN, natural_ZN, nat_yn_clipped, nat_zn_clipped = _load_and_normalize_natural_data(
        natural_data_path, y_mean, y_std, z_mean=z_mean, z_std=z_std
    )

    # === Opto light for B' training. The frozen opsin needs raw mW/mm² (model_fit z-scored it), but the
    # no-opsin subspace path applies B' to the light directly — and every controller feeds that path
    # z-scored light ((u - u_mean)/u_std, from lds.u_mean/u_std). So B' must be trained in the SAME gauge
    # it is deployed in: raw when use_opsin, z-scored otherwise. nat_u_ref is the input-free ("no light")
    # value in that gauge: raw 0, or its z-scored image -u_mean/u_std. ===
    opto_UN = UN[:, :k_opto]
    if use_opsin:
        opto_input = np.clip(opto_UN * u_std[:k_opto] + u_mean[:k_opto], 0, None).astype(np.float32)
        nat_u_ref = np.zeros(k_opto, dtype=np.float32)
    else:
        opto_input = opto_UN.astype(np.float32)
        nat_u_ref = (-u_mean[:k_opto] / u_std[:k_opto]).astype(np.float32)

    # === Concat NormStats over the augmented [LFP|behavior] output (width ny_lfp+nz). u is identity: the
    # light already carries its intended gauge (raw or z-scored per above), so normalize_bundle must not
    # rescale it. This is bundle.norm — the single source for the output bias d = -(y_mean/y_std). ===
    norm_stats = NormStats(
        y_mean=torch.tensor(np.concatenate([y_mean, z_mean]), dtype=torch.float32),
        y_std=torch.tensor(np.concatenate([y_std, z_std]), dtype=torch.float32),
        u_mean=torch.zeros(k_opto, dtype=torch.float32),
        u_std=torch.ones(k_opto, dtype=torch.float32),
    )

    bundle = load_cold_databundle(
        YN, ZN, opto_input, XN, natural_YN, natural_ZN, norm_stats, cfg, device,
        k_opto=k_opto, seed=seed, nat_u_ref=nat_u_ref,
    )

    # === Opsin checkpoint dir (loaded inside the pipeline with the 'external' provenance gate). Only
    # dereferenced when use_opsin; a no_opsin cell passes a placeholder the pipeline never opens. ===
    opsin_path = config.get('opsin_checkpoint_path')
    if use_opsin and not opsin_path:
        raise ValueError(
            "use_opsin=True but config['opsin_checkpoint_path'] is unset; model_fit.py should thread "
            "it from the opsin_train_cold sentinel (results/opsin_train_cold/opsin_stim_method_<m>/).")
    opsin_dir = Path(opsin_path) if opsin_path else Path(".")

    save_root = Path(str(outfiles.lds_model)).parent / "usid_training"

    log.info("Starting USID inline training via cold_full_run...")
    lds, Zhat, Yhat, Xhat = cold_full_run(cfg, bundle, norm_stats, opsin_dir, save_root, device)

    # Full resolved USID config -> model_fit folds it into the per-cell config_snapshot.yaml.
    lds.metadata['resolved_usid_config'] = _cfg_to_plain(cfg)
    # Natural-YN/ZN sanitize clip counts -> model_fit merges them with the Y/Z/U/X counts for summary.csv.
    lds.sanitize_clip_counts = {
        "Natural_YN": int(nat_yn_clipped),
        "Natural_ZN": int(nat_zn_clipped),
    }

    log.info("USID model fitting complete (inline training via cold_full_run)")
    return lds, Zhat, Yhat, Xhat


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Validation loss-weight regimes selected by the model_fit `val_regime` wildcard. The validation loss is
# the criterion for best-val checkpointing (both legs) and multistart rung-culling, so it steers which basin
# the fit keeps. Only delta/rollout differ between regimes; the leg-specific regularizers stay 0 at
# validation. `validation.rollout` (K/W) is left to cold_usid.yaml (this injects only loss_weights).
_VAL_REGIMES = {
    "delta_dom":   {"delta": 1.0, "rollout": 0.2, "teacher": 0.0, "innov_white": 0.0, "ifp": 0.0, "kreg": 0.0},
    "rollout_dom": {"delta": 0.2, "rollout": 1.0, "teacher": 0.0, "innov_white": 0.0, "ifp": 0.0, "kreg": 0.0},
}


def _load_usid_config(ny: int, k_opto: int, seed: int, cold_overrides: dict = None, nz: int = None,
                      fold_behavior: bool = True, natural_restarts: int = 1, usid_restarts: int = 1,
                      val_regime: str = "rollout_dom", pool_row_len: int = 800):
    """Load COLD-side USID config and apply runtime + COLD-config overrides.

    Base config is ``config/cold_usid.yaml`` (inside the COLD repo), maintained
    independently of USID's synthetic-training configs. Override stack, outer
    wins on conflicts:

      1. cold_usid.yaml (base).
      2. Runtime-dependent fields injected here (data.source, models.ny,
         models.n_clusters, models.nz, models.fold_behavior,
         multistart.natural_restarts, multistart.usid_restarts,
         validation.loss_weights (val_regime wildcard),
         usid_hyperparams.pool_row_len (pool_row_len wildcard), seed).
      3. COLD config's ``usid_overrides:`` block (if present) — lets individual
         COLD configs (e.g., ultra_light_config.yaml) tweak any USID hyperparam
         without editing cold_usid.yaml itself.

    fold_behavior comes from the model_fit fold_behavior wildcard (a real cond-param):
    True keeps the full-output observer, False pins the observer to the LFP block.
    """
    cold_usid_yaml = (
        Path(__file__).resolve().parent.parent.parent / "config" / "cold_usid.yaml"
    )
    if not cold_usid_yaml.exists():
        raise FileNotFoundError(
            f"COLD-side USID config not found at {cold_usid_yaml}. "
            "This file is tracked in the COLD repo at config/cold_usid.yaml; "
            "restore it from version control."
        )

    with open(cold_usid_yaml, "r", encoding="utf-8") as f:
        base_dict = yaml.safe_load(f) or {}

    if val_regime not in _VAL_REGIMES:
        raise ValueError(f"val_regime must be one of {sorted(_VAL_REGIMES)}, got {val_regime!r}")

    overrides = {
        "data": {
            "source": "cold_inline",
        },
        "models": {
            "ny": ny,            # LFP block width; the stages build the model at ny+nz output
            "n_clusters": k_opto,
            "nz": nz,            # behavior partition width (ZN.shape[1]) of the merged readout Cy
            "fold_behavior": fold_behavior,   # observer fold (fold_behavior wildcard); False -> LFP-only observer
        },
        # Per-leg counts from the model_fit natural_restarts/usid_restarts wildcards, deep-merged with
        # cold_usid.yaml's shared rungs/rule/eta/grace/base_seed. The usid leg is the seed-lottery stage.
        "multistart": {"natural_restarts": natural_restarts, "usid_restarts": usid_restarts},
        # Validation loss-weights from the val_regime wildcard (selection criterion for best-val + multistart
        # culling); overrides cold_usid.yaml's validation.loss_weights, leaving validation.rollout untouched.
        "validation": {"loss_weights": _VAL_REGIMES[val_regime]},
        # Opto-pool row length from the model_fit pool_row_len wildcard; sets how long a row the frozen
        # opsin GRU free-runs and how many windows each row yields.
        "usid_hyperparams": {"pool_row_len": pool_row_len},
        "seed": seed,
    }
    merged_dict = _deep_merge(base_dict, overrides)
    if cold_overrides:
        merged_dict = _deep_merge(merged_dict, cold_overrides)
    return to_ns(merged_dict)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict (override wins on conflicts)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_and_normalize_natural_data(
    natural_data_path: str,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z_mean: np.ndarray = None,
    z_std: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load natural electrode data, normalize with model_fit stats, and
    sanitize post-normalization overflow.

    Normalization uses the sys_id-derived (y_mean, y_std) so the natural
    and sys_id arrays share a coordinate system. Sanitization via the
    shared `sanitize_normalized` helper keeps this in sync with
    `model_fit.py`'s YN/ZN/UN/XN handling — same ±1e6 clip, same warning
    format. Cast path is float64 → clip → float32 so raw overflow doesn't
    pre-populate ±inf before the sanitizer can act.

    When (z_mean, z_std) are provided (behavior supplied), also load and normalize the natural
    behavior ``natural_expt.Z`` in the controller's Z coordinates — it becomes the behavior block of
    the augmented natural output, and the natural phase learns the Cy1 behavior rows against it.
    Returns natural_ZN=None when z-stats are absent (the no-behavior path never needs it).

    Returns:
        Tuple of (normalized natural_YN, normalized natural_ZN or None, YN clip count, ZN clip count).
        Both clip counts feed summary.csv; the ZN count is 0 on the no-behavior path.
    """
    log.info(f"Loading natural electrode data from: {natural_data_path}")
    natural_expt = ExptData.load(natural_data_path)
    natural_Y = natural_expt.Y.astype(np.float64)

    natural_YN = (natural_Y - y_mean) / y_std
    natural_YN, n_clipped = sanitize_normalized(natural_YN, "Natural_YN", log=log)
    natural_YN = natural_YN.astype(np.float32)

    natural_ZN = None
    nat_zn_clipped = 0
    if z_mean is not None and z_std is not None:
        natural_Z = drop_constant_behavior(natural_expt.Z, "Natural_Z", log=log).astype(np.float64)
        natural_ZN = (natural_Z - z_mean) / z_std
        natural_ZN, nat_zn_clipped = sanitize_normalized(natural_ZN, "Natural_ZN", log=log)
        natural_ZN = natural_ZN.astype(np.float32)
        log.info(f"Natural behavior: T={natural_ZN.shape[0]}, nz={natural_ZN.shape[1]} "
                 f"(natural Z = behavior block of the augmented output; the Cy1 behavior rows are "
                 f"learned against it in the natural phase)")

    log.info(
        f"Natural data: T={natural_YN.shape[0]}, ny={natural_YN.shape[1]}, "
        f"normalized with model_fit stats"
    )
    return natural_YN, natural_ZN, n_clipped, nat_zn_clipped

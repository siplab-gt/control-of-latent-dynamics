"""
USID model fitting for control-of-latent-dynamics pipeline.

Trains USID inline (natural model + USID model) using the same data that
model_fit.py normalizes, then returns a SubspaceLDS that the Controller
consumes directly. SubspaceLDS exposes unified A, B, C, Q, R properties
(computed from its subspace matrices) that satisfy the LDS interface.

This mirrors the BRAID pattern: fit_lds_to_data receives already-normalized
data, trains the model, and returns an LDS-like object.  Normalization
statistics are stamped onto it by model_fit.py after return.

Usage in Snakefile:
    Set config["model_fit_method"] = "usid" to use this module.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Ridge

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

from scripts.usid_subspace_interface import SubspaceLDS, usid_to_subspace_lds
from scripts.usid_training_core import run_training
from scripts.usid_data_interface import (
    DataBundle,
    slice_trajectory,
    split_sequence_list,
    stack_sequences_for_multisample,
)
from scripts.utils.config_io import to_ns
from scripts.utils.checkpoints import load_trained_opsin
from scripts.diagnostics import run_shared_diagnostics, run_external_diagnostics
from src.scripts.sys_id_expts import ExptData


def fit_lds_to_data(
    YN: np.ndarray,
    ZN: np.ndarray,
    UN: np.ndarray,
    X_fsnn: np.ndarray,
    n_x: int,
    cond_params: dict,
    base_params: dict,
    outfiles,
    config: dict,
):
    """
    Train USID model inline and return SubspaceLDS for MPC control.

    SubspaceLDS preserves the natural/unnatural state decomposition while
    exposing unified A, B, C properties that the standard
    LQMPCController consumes. This eliminates the intermediate LDS conversion.

    Steps:
    1. Load USID config from merged YAML, apply COLD overrides
    2. Load and normalize natural electrode data using model_fit stats
    3. Prepare DataBundle from normalized sys_id + natural data
    4. Train Natural + USID models via run_training()
    5. Extract SubspaceLDS from trained USID model
    6. Fit behavioral decoder (Cz)
    7. Return SubspaceLDS with predictions

    Args:
        YN: (T, m) Observations (normalized by model_fit.py)
        ZN: (T, m_z) Behavioral outputs (normalized)
        UN: (T, k) Inputs (normalized, optogenetic + task)
        X_fsnn: (T, n_fsnn) Ground-truth fSNN latent factors (normalized)
        n_x: Target state dimension
        cond_params: Condition-specific parameters (seed, k, etc.)
        base_params: Base experiment parameters
        outfiles: Output file paths
        config: Experiment configuration (includes y_mean/y_std/u_mean/u_std,
                natural_data_path from model_fit.py)

    Returns:
        subspace_lds: SubspaceLDS (duck-types as LDS for Controller)
        Zhat: (T, m_z) Predicted behavioral outputs
        Yhat: (T, m) Predicted observations
        Xhat: (T, n_x) Estimated states
    """
    # Set device
    use_cpu = config.get("model_fit_device", "cpu").lower() == "cpu"
    device = "cpu" if use_cpu else "cuda"
    log.info(f"Using {device} for USID inline training")

    # Get dimensions from data
    T, ny = YN.shape
    nz = ZN.shape[1]
    k = UN.shape[1]
    k_opto = int(cond_params.get("k", k))

    # USID trains only on opto inputs (B1', B2' map light→state). Task input
    # coupling (B_task) would require a separate regression or joint training
    # approach that hasn't been implemented yet.
    if k > k_opto:
        raise NotImplementedError(
            f"USID model_fit does not support model_task_inputs=True "
            f"(UN has {k} columns but k_opto={k_opto}). "
            f"Task input coupling is not implemented for USID."
        )

    # Seed for reproducibility
    seed = int(cond_params.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # === 1. Load USID config and apply COLD overrides ===
    cfg = _load_usid_config(ny, k_opto, seed, config.get("usid_overrides"))

    # === 2. Load and normalize natural electrode data ===
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

    natural_YN = _load_and_normalize_natural_data(
        natural_data_path, y_mean, y_std
    )

    # === 2b. Optionally load opsin model ===
    use_opsin = config.get('use_opsin', False)
    opsin_model = None
    if use_opsin:
        opsin_path = config.get('opsin_checkpoint_path')
        if opsin_path:
            opsin_results_dir = Path(opsin_path)
        else:
            opsin_results_dir = Path(USID_REPO_PATH) / "experiments" / "opsin_results"
        opsin_model = load_trained_opsin(
            opsin_results_dir, num_light_channels=1, map_location=device
        )
        if opsin_model is None:
            raise FileNotFoundError(
                f"use_opsin=True but no trained opsin checkpoint found in {opsin_results_dir}"
            )
        log.info(f"Loaded opsin model (input_scale={opsin_model.input_scale.item():.1f}, "
                 f"output_scale={opsin_model.output_scale.item():.2f})")

    # === 3. Prepare DataBundle ===
    opto_UN = UN[:, :k_opto]
    if use_opsin:
        # Un-zscore light so opsin model receives raw mW/mm² inputs.
        # model_fit.py z-scored everything; usid.py reverses it locally.
        raw_light = opto_UN * u_std[:k_opto] + u_mean[:k_opto]
        raw_light = np.clip(raw_light, 0, None)
        log.info(f"Un-zscored light for opsin: range [{raw_light.min():.1f}, {raw_light.max():.1f}] mW/mm²")
        opto_input = raw_light
    else:
        opto_input = opto_UN

    data_bundle = _prepare_data_bundle(
        YN, opto_input, natural_YN, ny, k_opto, cfg, seed,
        ZN=ZN, X_fsnn=X_fsnn, y_mean=y_mean, y_std=y_std,
    )

    # === 4. Train ===
    save_root = Path(str(outfiles.lds_model)).parent / "usid_training"
    save_root.mkdir(parents=True, exist_ok=True)

    log.info("Starting USID inline training...")
    result = run_training(
        cfg, data_bundle, save_root,
        trained_opsin_model=opsin_model,
        no_opsin=not use_opsin,
    )
    log.info(f"Training complete: n1={result.n1}, n2={result.n2}")

    # === 4b. Diagnostics ===
    try:
        run_shared_diagnostics(result, data_bundle, cfg, save_root)
        run_external_diagnostics(result, data_bundle, cfg, save_root)
    except Exception:
        import traceback
        traceback.print_exc()
        log.warning("Non-critical: diagnostics failed, continuing with LDS conversion")

    n1, n2 = result.n1, result.n2
    usid_model = result.usid_model

    # === 5. Extract SubspaceLDS ===
    log.info("Converting USID to SubspaceLDS...")
    subspace_lds = usid_to_subspace_lds(usid_model)

    # Compute predictions on full sys_id data
    log.info("Computing predictions on sys_id data...")
    y_tensor = torch.tensor(YN, dtype=torch.float32).unsqueeze(0).to(device)
    # Use raw light when opsin is active (model expects raw mW/mm² with no_opsin=False)
    u_for_pred = raw_light if use_opsin else UN[:, :k_opto]
    u_tensor = torch.tensor(u_for_pred, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = usid_model.forward(y_tensor, u_tensor)
        Yhat = y_pred.squeeze(0).cpu().numpy()

    Xhat = _compute_state_trajectory(usid_model, y_tensor, u_tensor, device)

    # === 6. Post-processing ===
    # Fit behavioral decoder
    log.info("Fitting behavioral decoder (Cz)...")
    Cz1, Cz2 = _fit_behavioral_decoder(Xhat, ZN, n1, n2)
    subspace_lds.Cz1 = Cz1
    subspace_lds.Cz2 = Cz2
    Zhat = Xhat[:, :n1] @ Cz1.T + Xhat[:, n1:] @ Cz2.T

    subspace_lds.metadata['k_opto'] = k_opto

    if use_opsin:
        subspace_lds.opsin_model = opsin_model.cpu()
        subspace_lds.metadata['use_opsin'] = True
        subspace_lds.metadata['max_light_mWmm2'] = float(
            config.get('u_max_mWmm2', base_params.get('u_max_mWmm2', 75.0))
            if isinstance(base_params, dict)
            else getattr(base_params, 'u_max_mWmm2', 75.0)
        )

    # Note: model_fit.py stamps normalization stats (y_mean, y_std, etc.) and
    # linreg_lds2fsnn onto the returned object after we return. The controller
    # uses those stats to zscore reference trajectories and state estimates.

    log.info("USID model fitting complete (inline training)")
    return subspace_lds, Zhat, Yhat, Xhat


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_usid_config(ny: int, k_opto: int, seed: int, cold_overrides: dict = None):
    """Load USID merged config and apply COLD-specific overrides.

    cold_overrides is an optional dict (from COLD config's ``usid_overrides``
    block) that is deep-merged on top of the built-in overrides, letting a COLD
    config tweak any USID hyperparam (e.g., epochs) without rebuilding the USID
    merged YAML.
    """
    merged_yaml_path = Path(USID_REPO_PATH) / "snake" / "build" / "usid_model.merged.yaml"
    if not merged_yaml_path.exists():
        raise FileNotFoundError(
            f"Merged USID config not found at {merged_yaml_path}. "
            "Run 'snakemake --snakefile snake/Snakefile --config experiment=usid_model' first."
        )

    with open(merged_yaml_path, "r", encoding="utf-8") as f:
        base_dict = yaml.safe_load(f) or {}

    overrides = {
        "data": {
            "source": "cold_inline",
        },
        "models": {
            "ny": ny,
            "n_clusters": k_opto,
        },
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
) -> np.ndarray:
    """Load natural electrode data and normalize with model_fit stats."""
    log.info(f"Loading natural electrode data from: {natural_data_path}")
    natural_expt = ExptData.load(natural_data_path)
    natural_Y = natural_expt.Y.astype(np.float32)

    natural_YN = (natural_Y - y_mean) / y_std
    log.info(
        f"Natural data: T={natural_YN.shape[0]}, ny={natural_YN.shape[1]}, "
        f"normalized with model_fit stats"
    )
    return natural_YN


def _prepare_data_bundle(
    opto_YN: np.ndarray,
    opto_UN: np.ndarray,
    natural_YN: np.ndarray,
    ny: int,
    k_opto: int,
    cfg,
    seed: int,
    ZN: np.ndarray = None,
    X_fsnn: np.ndarray = None,
    y_mean: np.ndarray = None,
    y_std: np.ndarray = None,
) -> DataBundle:
    """Slice and split normalized data into a DataBundle for run_training."""
    try:
        nat_seq_len = int(cfg.natural_hyperparams.window_seq_len)
        nat_stride = int(cfg.natural_hyperparams.window_stride)
        nat_train_ratio = float(cfg.natural_hyperparams.train_split)
        nat_val_ratio = float(cfg.natural_hyperparams.val_split)
        usid_seq_len = int(cfg.usid_hyperparams.window_seq_len)
        usid_stride = int(cfg.usid_hyperparams.window_stride)
        usid_train_ratio = float(cfg.usid_hyperparams.train_split)
        usid_val_ratio = float(cfg.usid_hyperparams.val_split)
    except AttributeError as e:
        raise AttributeError(
            "Missing required USID config attribute for data preparation."
        ) from e

    # --- Natural sequences ---
    natural_seqs = slice_trajectory(natural_YN, nat_seq_len, nat_stride)
    if len(natural_seqs) < 2:
        raise ValueError(
            f"Not enough natural sequences after slicing: {len(natural_seqs)} "
            f"(seq_len={nat_seq_len}, stride={nat_stride})"
        )

    nat_train, nat_val, _, _ = split_sequence_list(
        natural_seqs, nat_train_ratio, nat_val_ratio, "Natural", seed=seed
    )
    y_nat_train = stack_sequences_for_multisample(nat_train, ny, "Natural train")
    y_nat_val = stack_sequences_for_multisample(nat_val, ny, "Natural val")

    log.info(f"Natural: {len(nat_train)} train, {len(nat_val)} val sequences")

    # --- Opto sequences (with optional Z and X_fsnn) ---
    opto_y_seqs = slice_trajectory(opto_YN, usid_seq_len, usid_stride)
    opto_u_seqs = slice_trajectory(opto_UN, usid_seq_len, usid_stride)
    opto_z_seqs = slice_trajectory(ZN, usid_seq_len, usid_stride) if ZN is not None else None
    opto_x_seqs = slice_trajectory(X_fsnn, usid_seq_len, usid_stride) if X_fsnn is not None else None

    if len(opto_y_seqs) != len(opto_u_seqs):
        raise ValueError(
            f"Opto Y and U sequence counts differ: {len(opto_y_seqs)} vs {len(opto_u_seqs)}"
        )
    if len(opto_y_seqs) < 2:
        raise ValueError(
            f"Not enough opto sequences after slicing: {len(opto_y_seqs)} "
            f"(seq_len={usid_seq_len}, stride={usid_stride})"
        )

    # Build combined list for splitting (include Z and X_fsnn when available)
    combined = list(zip(
        opto_y_seqs,
        opto_u_seqs,
        opto_z_seqs if opto_z_seqs is not None else [None] * len(opto_y_seqs),
        opto_x_seqs if opto_x_seqs is not None else [None] * len(opto_y_seqs),
    ))
    train_combined, val_combined, _, _ = split_sequence_list(
        combined, usid_train_ratio, usid_val_ratio, "Opto", seed=seed
    )

    train_y_list = [c[0] for c in train_combined]
    train_u_list = [c[1] for c in train_combined]
    val_y_list = [c[0] for c in val_combined]
    val_u_list = [c[1] for c in val_combined]
    val_z_list = [c[2] for c in val_combined] if ZN is not None else None
    val_x_list = [c[3] for c in val_combined] if X_fsnn is not None else None

    y_usid_train = stack_sequences_for_multisample(train_y_list, ny, "USID train Y")
    u_usid_train = stack_sequences_for_multisample(train_u_list, k_opto, "USID train U")
    y_usid_val = stack_sequences_for_multisample(val_y_list, ny, "USID val Y")
    u_usid_val = stack_sequences_for_multisample(val_u_list, k_opto, "USID val U")

    log.info(f"Opto: {len(train_combined)} train, {len(val_combined)} val sequences")

    metadata = {
        'source': 'cold_inline',
        'usid_train_count': len(train_combined),
        'usid_val_count': len(val_combined),
        'nat_train_count': len(nat_train),
        'nat_val_count': len(nat_val),
    }
    if y_mean is not None:
        metadata['y_mean'] = y_mean
    if y_std is not None:
        metadata['y_std'] = y_std

    return DataBundle(
        train_y=y_usid_train,
        train_u=u_usid_train,
        val_y=y_usid_val,
        val_u=u_usid_val,
        natural_train_y=y_nat_train,
        natural_val_y=y_nat_val,
        ground_truth=None,
        val_y_seqs=val_y_list,
        val_u_seqs=val_u_list,
        val_source_types=['input_driven'] * len(val_y_list),
        train_source_types=['input_driven'] * len(train_y_list),
        val_z_seqs=val_z_list,
        val_reference_states_seqs=val_x_list,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Post-training helpers (compute trajectories, decoder, noise)
# ---------------------------------------------------------------------------

def _compute_state_trajectory(usid_model, y_train, u_train, device):
    """Compute state trajectory from USID model."""
    usid_model.eval()
    batch_size, T, ny = y_train.shape
    n1 = usid_model.n1
    n2 = usid_model.n2

    x1_traj = []
    x2_traj = []

    with torch.no_grad():
        with usid_model.natural_model.A_param.cache(), usid_model.A22_param.cache():
            photocurrents = usid_model.compute_photocurrents(u_train)

            x1t = torch.zeros(batch_size, n1, device=device)
            x2t = torch.zeros(batch_size, n2, device=device)

            for t in range(T):
                x1_traj.append(x1t.clone())
                x2_traj.append(x2t.clone())
                x1t, x2t, _ = usid_model.step(
                    x1t, x2t, y_train[:, t, :], photocurrents[:, t, :]
                )

    x1_traj = torch.stack(x1_traj, dim=1).squeeze(0).cpu().numpy()
    x2_traj = torch.stack(x2_traj, dim=1).squeeze(0).cpu().numpy()

    return np.hstack([x1_traj, x2_traj])


def _fit_behavioral_decoder(Xhat, ZN, n1, n2, alpha=1.0):
    """Fit linear decoder from states to behavioral output Z.

    Uses fit_intercept=False because ZN is already zero-mean (normalized by
    model_fit.py). This ensures x2z(X) = X @ Cz.T with no bias term, matching
    the controller's expectation.
    """
    decoder = Ridge(alpha=alpha, fit_intercept=False)
    decoder.fit(Xhat, ZN)

    Cz_combined = decoder.coef_  # (nz, n1+n2)
    Cz1 = Cz_combined[:, :n1]
    Cz2 = Cz_combined[:, n1:]

    return Cz1, Cz2



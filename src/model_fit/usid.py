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
from src.helpers import sanitize_normalized
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

    natural_YN, nat_yn_clipped = _load_and_normalize_natural_data(
        natural_data_path, y_mean, y_std
    )

    # === 2b. Optionally load opsin model ===
    use_opsin = bool(config['use_opsin'])
    opsin_model = None
    if use_opsin:
        opsin_path = config['opsin_checkpoint_path']
        if opsin_path:
            opsin_results_dir = Path(opsin_path)
        else:
            opsin_results_dir = Path(USID_REPO_PATH) / "experiments" / "opsin_results"
        # v3.2: COLD reuses the USID-synthetic NN opsin unchanged. The v3
        # `UnitlessChr2FourState` plant shares cleo's stock chr2_4s kinetic
        # rate constants and equation form; their photocurrents for any
        # given light input differ only by a per-channel scalar factor
        # absorbed into USID's learned B'. See plan §4.5 and justification
        # §4.5 for the compositional argument. load_trained_opsin just
        # picks the newest timestamp subdirectory — no COLD/USID namespace
        # separation.
        opsin_model = load_trained_opsin(
            opsin_results_dir,
            num_light_channels=1,
            map_location=device,
        )
        if opsin_model is None:
            raise FileNotFoundError(
                f"use_opsin=True but no trained opsin checkpoint found in "
                f"{opsin_results_dir}. Run USID synthetic opsin training "
                f"(scripts/opsin_model_testing.py) at least once; COLD "
                f"loads the latest plain-timestamp checkpoint."
            )
        log.info(f"Loaded opsin model (input_scale={opsin_model.input_scale.item():.1f}, "
                 f"output_scale={opsin_model.output_scale.item():.2f})")

    # === 3. Prepare DataBundle ===
    opto_UN = UN[:, :k_opto]
    channel_neuron_transmittance = config.get('channel_neuron_transmittance')
    if use_opsin:
        # Un-zscore light so opsin model receives raw mW/mm² inputs.
        # model_fit.py z-scored everything; usid.py reverses it locally.
        raw_light = opto_UN * u_std[:k_opto] + u_mean[:k_opto]
        raw_light = np.clip(raw_light, 0, None)
        log.info(f"Un-zscored light for opsin: range [{raw_light.min():.1f}, {raw_light.max():.1f}] mW/mm²")
        opto_input = raw_light
    else:
        opto_input = opto_UN

    # "No-input equivalent" for natural_electrode_data when it enters USID
    # training as input-free samples. Must match opto_input's space so the
    # USID model's forward pass sees a consistent representation:
    #   * use_opsin=True   → opto_input is raw mW/mm²   → nat u = 0.0
    #   * use_opsin=False  → opto_input is z-scored     → nat u = zscore(0)
    #                                                    = -u_mean/u_std  (per channel)
    # Shape: (k_opto,) — broadcast to (T, k_opto) inside _prepare_data_bundle.
    if use_opsin:
        nat_zero_u_ref = np.zeros(k_opto, dtype=np.float32)
    else:
        # u_std comes from model_fit.py's sanitize_normalized-backed _normalize,
        # which clamps zero/non-finite channels to 1.0 before writing
        # config['u_std']. So division here is safe by upstream invariant.
        # Explicit assertion for future refactor safety.
        assert np.all(np.isfinite(u_std[:k_opto])) and np.all(u_std[:k_opto] > 0), (
            "u_std must be finite and positive — guaranteed by model_fit.py's "
            "safe_std_compute. If this fires, the invariant is broken upstream."
        )
        nat_zero_u_ref = (-u_mean[:k_opto] / u_std[:k_opto]).astype(np.float32)

    # Opsin-fit validation inputs (present only when sys_id_expts recorded Iopto).
    # The diagnostic always runs in raw-channel-light space + transmittance —
    # this lets it score per-neuron irradiance vs bio at representative neurons
    # regardless of whether USID training uses cluster-irradiance or per-channel.
    Iopto_per_neuron = config.get('Iopto_per_neuron')
    if Iopto_per_neuron is not None:
        raw_light_all = opto_UN * u_std[:k_opto] + u_mean[:k_opto]
        raw_light_all = np.clip(raw_light_all, 0, None)
    else:
        raw_light_all = None

    data_bundle = _prepare_data_bundle(
        YN, opto_input, natural_YN, ny, k_opto, cfg, seed,
        ZN=ZN, X_fsnn=X_fsnn, y_mean=y_mean, y_std=y_std,
        Iopto_per_neuron=Iopto_per_neuron,
        channel_neuron_transmittance=channel_neuron_transmittance,
        raw_light_all=raw_light_all,
        nat_zero_u_ref=nat_zero_u_ref,
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
    # Same input space as training: opto_input is raw mW/mm² when use_opsin,
    # z-scored light when not.
    u_for_pred = opto_input if use_opsin else UN[:, :k_opto]
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
        subspace_lds.metadata['max_light_mWmm2'] = float(config['expts']['u_max_mWmm2'])

    # Stash natural-YN sanitize clip count so model_fit.py can merge it into
    # lds.sanitize_clip_counts → analyze_model_fit → summary.csv. Y/Z/U/X
    # counts are added in model_fit.py after we return.
    subspace_lds.sanitize_clip_counts = {"Natural_YN": int(nat_yn_clipped)}

    # Note: model_fit.py stamps normalization stats (y_mean, y_std, etc.) and
    # linreg_lds2fsnn onto the returned object after we return. The controller
    # uses those stats to zscore reference trajectories and state estimates.

    log.info("USID model fitting complete (inline training)")
    return subspace_lds, Zhat, Yhat, Xhat


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_usid_config(ny: int, k_opto: int, seed: int, cold_overrides: dict = None):
    """Load COLD-side USID config and apply runtime + COLD-config overrides.

    Base config is ``config/cold_usid.yaml`` (inside the COLD repo), maintained
    independently of USID's synthetic-training configs. Override stack, outer
    wins on conflicts:

      1. cold_usid.yaml (base).
      2. Runtime-dependent fields injected here (data.source, models.ny,
         models.n_clusters, seed).
      3. COLD config's ``usid_overrides:`` block (if present) — lets individual
         COLD configs (e.g., ultra_light_config.yaml) tweak any USID hyperparam
         without editing cold_usid.yaml itself.

    Historical note: previously loaded ``snake/build/usid_model.merged.yaml``
    from the USID repo, which coupled COLD to USID's synthetic Snakemake
    pipeline having been run. That dependency is gone.
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
) -> tuple[np.ndarray, int]:
    """Load natural electrode data, normalize with model_fit stats, and
    sanitize post-normalization overflow.

    Normalization uses the sys_id-derived (y_mean, y_std) so the natural
    and sys_id arrays share a coordinate system. Sanitization via the
    shared `sanitize_normalized` helper keeps this in sync with
    `model_fit.py`'s YN/ZN/UN/XN handling — same ±1e6 clip, same warning
    format. Cast path is float64 → clip → float32 so raw overflow doesn't
    pre-populate ±inf before the sanitizer can act.

    Returns:
        Tuple of (normalized natural_YN, clip count for summary.csv).
    """
    log.info(f"Loading natural electrode data from: {natural_data_path}")
    natural_expt = ExptData.load(natural_data_path)
    natural_Y = natural_expt.Y.astype(np.float64)

    natural_YN = (natural_Y - y_mean) / y_std
    natural_YN, n_clipped = sanitize_normalized(natural_YN, "Natural_YN", log=log)
    natural_YN = natural_YN.astype(np.float32)

    log.info(
        f"Natural data: T={natural_YN.shape[0]}, ny={natural_YN.shape[1]}, "
        f"normalized with model_fit stats"
    )
    return natural_YN, n_clipped


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
    Iopto_per_neuron: np.ndarray = None,
    channel_neuron_transmittance: np.ndarray = None,
    raw_light_all: np.ndarray = None,
    nat_zero_u_ref: np.ndarray = None,
) -> DataBundle:
    """Slice and split normalized data into a DataBundle for run_training.

    The USID-phase training pool is built to MATCH synthetic's pattern:
    input-free (natural-electrode) samples are interleaved with input-driven
    (opto) samples at a ratio controlled by
    ``cfg.usid_hyperparams.input_free_data_split``. Input-free displaces
    input-driven 1:1 so total USID pool size stays at N_mixed (opto windows).
    See scripts/usid_data_interface.py:371-417 for the synthetic reference.

    Args:
        nat_zero_u_ref: (k_opto,) per-channel "no-input equivalent" value.
            Broadcast to (usid_seq_len, k_opto) for each natural sample's u.
            Zeros when opto_UN is in raw mW/mm² space; -u_mean/u_std when
            opto_UN is z-scored. Computed by the caller (fit_lds_to_data).
            None disables input-free injection (falls back to opto-only).
    """
    try:
        nat_seq_len = int(cfg.natural_hyperparams.window_seq_len)
        nat_stride = int(cfg.natural_hyperparams.window_stride)
        nat_train_ratio = float(cfg.natural_hyperparams.train_split)
        # nat_val_ratio is NOT consulted. The natural phase only splits
        # train/val (no test), and val = 1 - nat_train_ratio of the
        # timeline — whatever nat_train doesn't claim. Synthetic's
        # reference (scripts/usid_data_interface.py:157-202) also has
        # only train/val for the natural phase. The config key
        # `natural_hyperparams.val_split` is ignored here; we derive the
        # val share from the train ratio alone in both branches below so
        # the two paths stay consistent.
        nat_val_ratio_derived = 1.0 - nat_train_ratio
        usid_seq_len = int(cfg.usid_hyperparams.window_seq_len)
        usid_stride = int(cfg.usid_hyperparams.window_stride)
        usid_train_ratio = float(cfg.usid_hyperparams.train_split)
        usid_val_ratio = float(cfg.usid_hyperparams.val_split)
        input_free_split = float(cfg.usid_hyperparams.input_free_data_split)
        input_free_split = min(max(input_free_split, 0.0), 1.0)
    except AttributeError as e:
        raise AttributeError(
            "Missing required USID config attribute for data preparation."
        ) from e

    # --- Time-split natural_YN ONCE into train/val halves, then slice each
    #     half independently at both nat_seq_len (for Natural phase) and
    #     usid_seq_len (for USID input-free). Guarantees timestamps in
    #     Natural-phase-val never leak into USID-phase-train-input-free
    #     (and vice-versa). Synthetic USID avoids this overlap by using
    #     separately-generated `states_input_free`/`outputs_input_free`
    #     arrays; COLD has only one natural recording, so a contiguous
    #     time-split is the structural analog.
    T_nat = natural_YN.shape[0]
    # Boundary uses nat_train_ratio; val goes until T_nat.
    nat_boundary = int(nat_train_ratio * T_nat)
    # slice_trajectory yields ≥1 window when T >= seq_len. Use `< nat_seq_len`
    # (not `< nat_seq_len + 1`) so a boundary that produces exactly one
    # window on each side still qualifies for the clean time-split path.
    if nat_boundary < nat_seq_len or (T_nat - nat_boundary) < nat_seq_len:
        # Fall back to the old random-window split if data is too short for
        # a clean time-split at nat_seq_len.
        log.warning(
            f"Natural data too short for time-split at nat_seq_len={nat_seq_len}: "
            f"T_nat={T_nat}, boundary={nat_boundary}. Falling back to random "
            f"window-level split (shared natural_YN across phases, mild "
            f"train/val-leakage across phases)."
        )
        natural_seqs = slice_trajectory(natural_YN, nat_seq_len, nat_stride)
        if len(natural_seqs) < 2:
            raise ValueError(
                f"Not enough natural sequences after slicing: {len(natural_seqs)}"
            )
        nat_train, nat_val, _, _ = split_sequence_list(
            natural_seqs, nat_train_ratio, nat_val_ratio_derived, "Natural", seed=seed
        )
        nat_train_raw = natural_YN  # reused for USID input-free; leakage accepted
        nat_val_raw = None
        log.info(
            f"Natural phase (fallback random split): {len(nat_train)} train, "
            f"{len(nat_val)} val windows"
        )
    else:
        nat_train_raw = natural_YN[:nat_boundary]
        nat_val_raw = natural_YN[nat_boundary:]
        nat_train = slice_trajectory(nat_train_raw, nat_seq_len, nat_stride)
        nat_val = slice_trajectory(nat_val_raw, nat_seq_len, nat_stride)
        if len(nat_train) < 1 or len(nat_val) < 1:
            raise ValueError(
                f"Natural time-split produced empty side: "
                f"train={len(nat_train)}, val={len(nat_val)} "
                f"(boundary={nat_boundary}, seq_len={nat_seq_len})."
            )
        log.info(
            f"Natural phase (time-split at t={nat_boundary}/{T_nat}): "
            f"{len(nat_train)} train, {len(nat_val)} val windows"
        )

    y_nat_train = stack_sequences_for_multisample(nat_train, ny, "Natural train")
    y_nat_val = stack_sequences_for_multisample(nat_val, ny, "Natural val")

    # --- Opto sequences for USID phase (input_driven) ---
    opto_y_seqs = slice_trajectory(opto_YN, usid_seq_len, usid_stride)
    opto_u_seqs = slice_trajectory(opto_UN, usid_seq_len, usid_stride)
    opto_z_seqs = (slice_trajectory(ZN, usid_seq_len, usid_stride)
                   if ZN is not None else None)
    opto_x_seqs = (slice_trajectory(X_fsnn, usid_seq_len, usid_stride)
                   if X_fsnn is not None else None)
    opto_iopto_seqs = (
        slice_trajectory(Iopto_per_neuron, usid_seq_len, usid_stride)
        if Iopto_per_neuron is not None else None
    )
    opto_light_raw_seqs = (
        slice_trajectory(raw_light_all, usid_seq_len, usid_stride)
        if raw_light_all is not None else None
    )

    if len(opto_y_seqs) != len(opto_u_seqs):
        raise ValueError(
            f"Opto Y and U sequence counts differ: {len(opto_y_seqs)} vs {len(opto_u_seqs)}"
        )
    if len(opto_y_seqs) < 2:
        raise ValueError(
            f"USID phase has fewer than 2 opto windows after slicing: "
            f"{len(opto_y_seqs)} (seq_len={usid_seq_len}, stride={usid_stride}, "
            f"T_opto={opto_YN.shape[0]}). "
            f"Need T_opto >= usid_seq_len + usid_stride. "
            f"Either increase sys_id trial counts "
            f"(configs.*.sys_id_expts.n_trials_sys_id) or lower usid_seq_len/"
            f"usid_stride in cold_usid.yaml."
        )

    N_mixed = len(opto_y_seqs)

    # --- Natural sequences for the USID phase (input_free). Sliced from
    #     nat_train_raw only (the time-split train half) so no USID
    #     input-free samples overlap timestamps in Natural-phase-val.
    if nat_val_raw is None:
        # Time-split fallback above — accept mild overlap, slice whole natural_YN
        nat_usid_y_seqs = slice_trajectory(natural_YN, usid_seq_len, usid_stride)
    else:
        nat_usid_y_seqs = slice_trajectory(nat_train_raw, usid_seq_len, usid_stride)
    N_nat_avail = len(nat_usid_y_seqs)

    # Ratio-driven target (matches scripts/usid_data_interface.py:371-386).
    # Input-free DISPLACES input-driven 1:1 so total pool stays N_mixed.
    #
    # Unlike synthetic, COLD does NOT pad with `extra_nat` when
    # mixed_target hits zero (i.e. input_free_split → 1.0) — pool shrinks
    # if N_nat_avail < N_mixed. Acceptable because COLD uses
    # input_free_split ≤ 0.5 in practice; the 100%-input-free corner case
    # isn't meaningful (no opto to learn B' against).
    if input_free_split <= 0.0 or N_nat_avail == 0 or nat_zero_u_ref is None:
        nat_target = 0
        mixed_target = N_mixed
    else:
        nat_target = int(round(input_free_split * N_mixed))
        nat_target = min(nat_target, N_nat_avail)
        mixed_target = N_mixed - nat_target
        # Edge case: config asks for > 0 nat but rounds to 0 — floor at 1 so
        # IFP can at least fire on a minimal sample.
        if nat_target == 0 and N_nat_avail > 0:
            nat_target = 1
            mixed_target = max(0, N_mixed - 1)

    # Build natural input-free entries (u = broadcast nat_zero_u_ref).
    nat_combined = []
    if nat_target > 0:
        nat_u_window = np.broadcast_to(
            nat_zero_u_ref[None, :], (usid_seq_len, k_opto)
        ).astype(np.float32)
        for i in range(nat_target):
            nat_combined.append((
                nat_usid_y_seqs[i],    # y: real natural-electrode LFP at u=0
                nat_u_window.copy(),   # u: no-input equivalent
                None,                  # z: no behavior target for input_free
                None,                  # x_fsnn: no reference factors
                None,                  # Iopto_per_neuron: none under u=0
                None,                  # raw_light: none under u=0
            ))

    # Build opto input-driven entries (truncated to mixed_target when displaced).
    N_opto_kept = mixed_target
    opto_combined = []
    for i in range(N_opto_kept):
        opto_combined.append((
            opto_y_seqs[i],
            opto_u_seqs[i],
            opto_z_seqs[i] if opto_z_seqs is not None else None,
            opto_x_seqs[i] if opto_x_seqs is not None else None,
            opto_iopto_seqs[i] if opto_iopto_seqs is not None else None,
            opto_light_raw_seqs[i] if opto_light_raw_seqs is not None else None,
        ))

    # Combined pool with parallel source_types list.
    # Order: input_free first (indices [0, nat_target)), then input_driven
    # (indices [nat_target, pool_size)). Keeps the split deterministic.
    combined = nat_combined + opto_combined
    pool_source = ['input_free'] * len(nat_combined) + ['input_driven'] * len(opto_combined)

    if len(combined) < 2:
        raise ValueError(
            f"USID pool has fewer than 2 sequences after input-free displacement "
            f"(nat_target={nat_target}, mixed_target={mixed_target})."
        )

    # Index-based train/val split preserving source_types (matches synthetic
    # usid_data_interface.py:430-456 — shuffled index split, not positional).
    rng_usid = np.random.default_rng(seed)
    pool_idx = np.arange(len(combined))
    rng_usid.shuffle(pool_idx)

    train_count = max(1, int(round(usid_train_ratio * len(combined))))
    if train_count >= len(combined):
        train_count = len(combined) - 1
    remaining = len(combined) - train_count
    val_count = int(round(usid_val_ratio * len(combined)))
    val_count = max(1, min(remaining, val_count))

    train_idx = pool_idx[:train_count]
    val_idx = pool_idx[train_count:train_count + val_count]
    if val_idx.size == 0:
        val_idx = pool_idx[train_count:train_count + 1]

    train_combined = [combined[i] for i in train_idx]
    val_combined = [combined[i] for i in val_idx]
    train_source_types = [pool_source[i] for i in train_idx]
    val_source_types = [pool_source[i] for i in val_idx]

    train_y_list = [c[0] for c in train_combined]
    train_u_list = [c[1] for c in train_combined]
    val_y_list = [c[0] for c in val_combined]
    val_u_list = [c[1] for c in val_combined]
    # val_z / val_x MAY contain None entries (input_free rows have no Z/X);
    # downstream consumers handle None per-sample via val_source_types.
    val_z_list = [c[2] for c in val_combined] if ZN is not None else None
    val_x_list = [c[3] for c in val_combined] if X_fsnn is not None else None

    # val_biophys_iopto / val_light_raw are consumed ONLY by the opsin-fit
    # diagnostic (NN-vs-biophysical photocurrent comparison), which is
    # meaningful ONLY on input-driven samples (input-free rows have u=0 with
    # no opto to compare against). Filter here so the diagnostic can
    # np.stack the lists without mixing None entries. Mirrors synthetic's
    # `driven_start_idx`-based masking in usid_data_interface.py:457-460.
    driven_mask = [s == 'input_driven' for s in val_source_types]
    if opto_iopto_seqs is not None:
        val_iopto_list = [c[4] for c, d in zip(val_combined, driven_mask) if d]
    else:
        val_iopto_list = None
    if opto_light_raw_seqs is not None:
        val_light_raw_list = [c[5] for c, d in zip(val_combined, driven_mask) if d]
    else:
        val_light_raw_list = None

    y_usid_train = stack_sequences_for_multisample(train_y_list, ny, "USID train Y")
    u_usid_train = stack_sequences_for_multisample(train_u_list, k_opto, "USID train U")
    y_usid_val = stack_sequences_for_multisample(val_y_list, ny, "USID val Y")
    u_usid_val = stack_sequences_for_multisample(val_u_list, k_opto, "USID val U")

    n_train_if = sum(1 for s in train_source_types if s == 'input_free')
    n_train_id = len(train_source_types) - n_train_if
    n_val_if = sum(1 for s in val_source_types if s == 'input_free')
    n_val_id = len(val_source_types) - n_val_if
    log.info(
        f"USID phase pool: {len(combined)} sequences "
        f"(input_free={nat_target}/{N_nat_avail} avail, input_driven={mixed_target}/{N_mixed} avail, "
        f"split={input_free_split:.2f}). "
        f"train: {n_train_id} input_driven + {n_train_if} input_free; "
        f"val: {n_val_id} input_driven + {n_val_if} input_free."
    )

    metadata = {
        'source': 'cold_inline',
        'usid_train_count': len(train_combined),
        'usid_val_count': len(val_combined),
        'usid_train_input_free': n_train_if,
        'usid_train_input_driven': n_train_id,
        'usid_val_input_free': n_val_if,
        'usid_val_input_driven': n_val_id,
        'usid_input_free_split': input_free_split,
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
        val_source_types=val_source_types,
        train_source_types=train_source_types,
        val_z_seqs=val_z_list,
        val_reference_states_seqs=val_x_list,
        val_biophys_iopto_seqs=val_iopto_list,
        val_light_raw_seqs=val_light_raw_list,
        channel_neuron_transmittance=channel_neuron_transmittance,
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



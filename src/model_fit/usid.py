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
import copy
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
        # COLD trains its own opsin NN against the calibrated
        # UnitlessChr2FourState plant via the ``opsin_train_cold`` snake rule.
        # The cold-trained checkpoint lands under
        # ``$USID_REPO_ROOT/experiments/opsin_results_external/cold/`` with
        # provenance.tag == 'external' and caller_tag == 'cold'. The
        # ``expected_provenance='external'`` gate refuses to silently load a
        # synthetic-tagged checkpoint if anything mis-wires the path back to
        # the synthetic tree — the whole point of the cold-train pipeline is
        # to remove the train-vs-deploy distribution mismatch the synthetic
        # NN was creating.
        #
        # Override path: callers can still pin ``opsin_checkpoint_path`` in
        # config to a specific dir. That overrides discovery but is still
        # gated by ``expected_provenance='external'`` — synthetic-tagged
        # checkpoints are refused at this seam.
        opsin_path = config.get('opsin_checkpoint_path')
        if opsin_path:
            opsin_results_dir = Path(opsin_path)
        else:
            opsin_results_dir = (
                Path(USID_REPO_PATH) / "experiments" / "opsin_results_external" / "cold"
            )
        opsin_model = load_trained_opsin(
            opsin_results_dir,
            num_light_channels=1,
            map_location=device,
            expected_provenance='external',
        )
        if opsin_model is None:
            raise FileNotFoundError(
                f"use_opsin=True but no cold-trained opsin checkpoint found in "
                f"{opsin_results_dir}. The ``opsin_train_cold`` snake rule "
                f"should have produced one; check its log."
            )
        log.info(f"Loaded opsin model (input_scale={opsin_model.input_scale.item():.1f}, "
                 f"output_scale={opsin_model.output_scale.item():.4f})")

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
    # raw_light_all and channel_neuron_transmittance are handed to the USID-side
    # diagnostic; transmittance is used there ONLY for rep-neuron selection
    # (argmax dominance), not to synthesize NN input. See
    # scripts/diagnostics.py::_run_opsin_fit_diagnostic for the scoring method
    # and the scratchpad/opsin_v4_plan.md §5.3 framing note.
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
    try:
        n_restarts = int(cfg.usid_hyperparams.n_restarts)
    except AttributeError as e:
        raise AttributeError(
            "Missing required USID config attribute usid_hyperparams.n_restarts."
        ) from e
    if n_restarts <= 1:
        result = run_training(
            cfg, data_bundle, save_root,
            trained_opsin_model=opsin_model,
            no_opsin=not use_opsin,
        )
    else:
        # Multistart: fit n_restarts independent models (distinct torch init seed,
        # same sys-id data) and keep the best by the configured selection metric.
        # The USID+opsin objective is multimodal (the seed lottery); multistart
        # escapes bad local optima. Selection metric (usid_hyperparams.multistart_selector):
        #   'freerun_r2y' (corrected default) — MAX held-out observer-OFF forced-path
        #     R2(Y): a self-fed free-run where opto reaches the output ONLY through B2',
        #     so it scores the control-relevant input path the Kalman observer cannot
        #     fake. For pin_bprime it is scored on a pinned deepcopy (the deployed-
        #     equivalent model); otherwise on the trained model as-is.
        #   'val_loss' — MIN held-out USID validation loss (observer-ON, Kalman-masked;
        #     ~uncorrelated with control Q — the prior, broken criterion).
        # Each restart writes its own save_root subdir; the selected result object
        # drives everything downstream (LDS handoff, model_fit metrics).
        try:
            multistart_selector = str(cfg.usid_hyperparams.multistart_selector)
        except AttributeError as e:
            raise AttributeError(
                "Missing required USID config attribute usid_hyperparams.multistart_selector."
            ) from e
        if multistart_selector not in ("freerun_r2y", "val_loss"):
            raise ValueError(
                f"Unknown multistart_selector={multistart_selector!r}; "
                "expected 'freerun_r2y' or 'val_loss'."
            )
        _ms_pin = bool(cfg.usid_hyperparams.pin_bprime)
        _ms_horizon = int(cfg.usid_hyperparams.cz_observer_horizon)
        y_tensor_ms = torch.tensor(YN, dtype=torch.float32).unsqueeze(0).to(device)
        u_for_pred_ms = opto_input if use_opsin else UN[:, :k_opto]
        u_tensor_ms = torch.tensor(u_for_pred_ms, dtype=torch.float32).unsqueeze(0).to(device)
        log.info(
            f"[multistart] {n_restarts} restarts; selector={multistart_selector} "
            f"(pin_bprime={_ms_pin}, horizon={_ms_horizon})"
        )
        candidates = []
        for i in range(n_restarts):
            torch.manual_seed(10007 * (i + 1))
            sr_i = save_root / f"restart_{i}"
            sr_i.mkdir(parents=True, exist_ok=True)
            r_i = run_training(
                cfg, data_bundle, sr_i,
                trained_opsin_model=opsin_model,
                no_opsin=not use_opsin,
            )
            v_loss = float(r_i.usid_val_hist[-1]) if r_i.usid_val_hist else float("inf")
            if multistart_selector == "val_loss":
                score = -v_loss          # argmax over score; lower val loss wins
                metric = v_loss
            else:  # freerun_r2y: held-out forced-path R2(Y) of the deployed-equivalent model
                Xh_i = _compute_state_trajectory(r_i.usid_model, y_tensor_ms, u_tensor_ms, device)
                if _ms_pin:
                    m_i = copy.deepcopy(r_i.usid_model)
                    r2y = _pin_bprime(m_i, Xh_i, YN, u_tensor_ms, device, _ms_horizon)["freerun_r2y_heldout"]
                else:
                    r2y = _freerun_r2y_heldout(r_i.usid_model, Xh_i, YN, u_tensor_ms, device, _ms_horizon)
                score = r2y if (r2y == r2y) else float("-inf")   # nan -> -inf
                metric = r2y
            log.info(f"[multistart] restart {i}: val_loss={v_loss:.6g}, "
                     f"{multistart_selector}={metric:.6g}")
            candidates.append((score, i, r_i, metric))
        _, i_sel, result, metric_sel = max(candidates, key=lambda c: c[0])
        log.info(
            f"[multistart] selected restart {i_sel}/{n_restarts} "
            f"(selector={multistart_selector}, metric={metric_sel:.6g}); "
            f"all={[(c[1], round(c[3], 6)) for c in candidates]}"
        )
    log.info(f"Training complete: n1={result.n1}, n2={result.n2}")

    # === 4b. Diagnostics (optional; never feed the LDS handoff or model_fit CSV) ===
    if cfg.diagnostics.enabled:
        try:
            run_shared_diagnostics(result, data_bundle, cfg, save_root)
            run_external_diagnostics(result, data_bundle, cfg, save_root)
        except Exception:
            import traceback
            traceback.print_exc()
            log.warning("Non-critical: diagnostics failed, continuing with LDS conversion")
    else:
        log.info("Diagnostics disabled (cfg.diagnostics.enabled=false); skipping plots")

    n1, n2 = result.n1, result.n2
    usid_model = result.usid_model

    # Compute predictions on full sys_id data (tensors reused by the pin + Cz fit).
    log.info("Computing predictions on sys_id data...")
    y_tensor = torch.tensor(YN, dtype=torch.float32).unsqueeze(0).to(device)
    # Same input space as training: opto_input is raw mW/mm² when use_opsin,
    # z-scored light when not.
    u_for_pred = opto_input if use_opsin else UN[:, :k_opto]
    u_tensor = torch.tensor(u_for_pred, dtype=torch.float32).unsqueeze(0).to(device)

    # Teacher-forced (filtered) state — the controller's running observer estimate;
    # reused as the pin/Cz free-run seeds.
    Xhat = _compute_state_trajectory(usid_model, y_tensor, u_tensor, device)

    # === 5b. Pin B' (optional) ===
    # Identify the opsin/forced input path B2' by a convex fit to the predictor
    # free-run output, decoupled from the Kalman-masked joint objective. Mutates
    # usid_model in place (B2' pinned, B1'=0), so the SubspaceLDS extraction, Cz fit,
    # and Yhat/Zhat below all see the pinned input map. See _pin_bprime.
    try:
        pin_bprime = bool(cfg.usid_hyperparams.pin_bprime)
    except AttributeError as e:
        raise AttributeError(
            "Missing required USID config attribute usid_hyperparams.pin_bprime."
        ) from e
    try:
        cz_fit_mode = str(cfg.usid_hyperparams.cz_fit_mode)
        cz_observer_horizon = int(cfg.usid_hyperparams.cz_observer_horizon)
    except AttributeError as e:
        raise AttributeError(
            "Missing required USID config attribute "
            "usid_hyperparams.cz_fit_mode / cz_observer_horizon."
        ) from e
    if pin_bprime:
        if not use_opsin:
            raise ValueError(
                "pin_bprime=True requires use_opsin=True (the pin identifies the "
                "opsin/forced input path B2'); got use_opsin=False."
            )
        log.info("Pinning B' via convex predictor-free-run fit...")
        pin_diag = _pin_bprime(usid_model, Xhat, YN, u_tensor, device, cz_observer_horizon)
        # Recompute the filtered trajectory with the pinned input map so the Cz fit
        # + Zhat decode from the post-pin state.
        Xhat = _compute_state_trajectory(usid_model, y_tensor, u_tensor, device)
    else:
        pin_diag = {"pin_bprime": False}

    with torch.no_grad():
        Yhat = usid_model.forward(y_tensor, u_tensor).squeeze(0).cpu().numpy()

    # === 5. Extract SubspaceLDS (reads the pinned input map) ===
    log.info("Converting USID to SubspaceLDS...")
    subspace_lds = usid_to_subspace_lds(usid_model)
    subspace_lds.metadata['pin_bprime'] = pin_diag

    # === 6. Post-processing ===
    # Fit behavioral decoder. The controller (NLMPC/iLQR) plans against the
    # OBSERVER free-run state (self-fed y = C·x), not the teacher-forced state
    # Xhat above — so by default Cz is fit on the pooled observer-rollout state
    # the controller visits, not on Xhat. See _fit_behavioral_decoder.
    log.info("Fitting behavioral decoder (Cz)...")
    Cz1, Cz2 = _fit_behavioral_decoder(
        Xhat, ZN, n1, n2,
        usid_model=usid_model,
        u_tensor=u_tensor,
        device=device,
        cz_fit_mode=cz_fit_mode,
        cz_observer_horizon=cz_observer_horizon,
    )
    subspace_lds.Cz1 = Cz1
    subspace_lds.Cz2 = Cz2
    # Zhat reported for the analysis CSV is the teacher-forced readout (the
    # state the model is validated against — what R2_model_fit_Z_Zhat measures).
    Zhat = Xhat[:, :n1] @ Cz1.T + Xhat[:, n1:] @ Cz2.T

    # Load-bearing controllability diagnostic: horizon-resolved free-run R²(Z).
    # R2_model_fit_Z_Zhat above is 1-step/filtered and decoupled from the open-loop
    # control the controller plans; this scores the predictor free-run the pin targets.
    # Reaches model_fit_analysis.csv via analyze_model_fit reading lds.metadata.
    subspace_lds.metadata['freerun_r2z'] = _freerun_r2z_by_horizon(
        usid_model, Xhat, ZN, u_tensor, Cz1, Cz2, device, cz_observer_horizon,
    )

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


def _observer_rollout_pool(usid_model, Xf, u_tensor, device, horizon):
    """Build the pooled OBSERVER free-run state the controller plans against.

    The NLMPC/iLQR controllers self-feed ``y = C·x`` (with the Kalman gains K1/K2
    still applied) when they roll the model forward to plan — see
    ``src/ctrl/NLMPC.py::_rollout_state``. This is NOT the teacher-forced
    trajectory ``Xf`` (from ``_compute_state_trajectory``), which feeds the REAL
    recorded y each step. Fitting Cz on the teacher-forced state therefore
    decodes Z from a state the controller never actually visits.

    Here we reproduce the controller's planning rollout exactly, reusing the
    trained model's own ``step`` (single source of the recurrence): from each
    filtered start ``Xf[t0]`` we free-run ``horizon`` steps, self-feeding the
    model's predicted y, and collect the state at every rollout depth
    ``j = 1..horizon`` paired with the true Z at ``t0 + j``. Pooling over all j
    (rather than fitting at one far horizon) yields one Cz that decodes Z
    consistently along the whole rollout — the near horizons (which the
    replanning MPC realizes most) stay intact while the tail is lifted.

    No trial boundaries are available at model_fit time, so rollouts are seeded
    from every start ``t0 = 0..T-1-horizon`` over the continuous trajectory; a
    rollout near a (concatenated) trial boundary spans it, which is benign
    because the photocurrent input is continuous.

    Args:
        Xf: (T, n1+n2) teacher-forced state trajectory (the controller's running
            observer estimate), reused as the rollout seeds.

    Returns:
        Tuple of (X_pool (n_starts*horizon, n1+n2), Z_idx (n_starts*horizon,)),
        where Z_idx maps each pooled state to its true-Z timestep.
    """
    usid_model.eval()
    n1 = usid_model.n1
    n2 = usid_model.n2
    T = Xf.shape[0]
    if T <= horizon + 1:
        raise ValueError(
            f"Sequence too short to build observer-rollout Cz pool: T={T}, "
            f"cz_observer_horizon={horizon}. Need T > horizon + 1."
        )

    Xf_t = torch.tensor(Xf, dtype=torch.float32, device=device)

    n_starts = T - horizon
    with torch.no_grad():
        with usid_model.natural_model.A_param.cache(), usid_model.A22_param.cache():
            # Photocurrents for the WHOLE trajectory (matches the GRU memory the
            # controller carries); index per-start below.
            photocurrents = usid_model.compute_photocurrents(u_tensor)  # (1, T, k)
            P = photocurrents.squeeze(0)  # (T, k)

            # Batch all starts: x1/x2 (n_starts, n1/n2); P broadcast per rollout step.
            x1 = Xf_t[:n_starts, :n1].clone()
            x2 = Xf_t[:n_starts, n1:].clone()
            start_idx = torch.arange(n_starts, device=device)

            X_pool = []
            Z_idx = []
            for j in range(horizon):
                # self-fed y = C1·x1 + C2·x2 (== model.step's y_out at this state)
                yhat = x1 @ usid_model.natural_model.Cy1.T + x2 @ usid_model.Cy2.T
                P_j = P[start_idx + j]  # photocurrent at absolute time t0+j
                x1, x2, _ = usid_model.step(x1, x2, yhat, P_j)
                # state at depth j+1 -> true Z at t0 + (j+1)
                X_pool.append(torch.cat([x1, x2], dim=1))
                Z_idx.append(start_idx + (j + 1))

    X_pool = torch.cat(X_pool, dim=0).cpu().numpy()
    Z_idx = torch.cat(Z_idx, dim=0).cpu().numpy()
    return X_pool, Z_idx


def _vw_r2(ytrue: torch.Tensor, ypred: torch.Tensor) -> float:
    """Variance-weighted R² matching sklearn's ``multioutput='variance_weighted'``.

    Equals 1 - Σ_i ss_res_i / Σ_i ss_tot_i over the outputs with non-zero variance
    (a constant output — e.g. the reaching task's 3rd Z dim — gets zero weight and is
    excluded, as sklearn does). Returns nan if every output is constant.
    """
    ss_res = ((ytrue - ypred) ** 2).sum(dim=0)
    ss_tot = ((ytrue - ytrue.mean(dim=0)) ** 2).sum(dim=0)
    mask = ss_tot > 0
    if not bool(mask.any()):
        return float("nan")
    return float(1.0 - ss_res[mask].sum() / ss_tot[mask].sum())


def _freerun_r2z_by_horizon(usid_model, Xhat, ZN, u_tensor, Cz1, Cz2, device, horizon):
    """Horizon-resolved free-run R²(Z) — the load-bearing controllability diagnostic.

    The 1-step teacher-forced ``R2_model_fit_Z_Zhat`` is systematically optimistic about
    control: it scores the filtered predictor (A+KC), feeding the real y each step, NOT
    the open-loop free-run the controller actually plans. This rolls the PREDICTOR
    free-run (self-fed y = C·x — ``USIDModel.predict(forced=False)`` /
    ``NLMPC._rollout_state``, the same recurrence the pin targets) from each filtered
    seed ``Xhat[t0]``, decodes Z = [x1,x2]·[Cz1,Cz2]ᵀ at each depth j, and reports
    variance-weighted R²(Z) vs the true Z at t0+j.

    h0 (the filtered readout at the seed) ≈ ``R2_model_fit_Z_Zhat`` on the windowed
    starts — a correctness gate. The value at the control horizon (h = ctrl_horizon_ms/dt,
    e.g. 10 for 100 ms, 20 for 200 ms) is what predicts whether the controller can track;
    ``predictable_horizon`` is the largest depth (contiguous from h0) with R²(Z) > 0.

    Returns a dict ``{f"h{j}": r2 for j in 0..horizon}`` plus ``predictable_horizon``.
    """
    n1, n2 = usid_model.n1, usid_model.n2
    T = Xhat.shape[0]
    horizon = int(min(horizon, T - 2))
    if horizon < 1:
        return {"h0": float("nan"), "predictable_horizon": 0}

    Cz1_t = torch.as_tensor(Cz1, dtype=torch.float32, device=device)
    Cz2_t = torch.as_tensor(Cz2, dtype=torch.float32, device=device)
    Xf_t = torch.as_tensor(Xhat, dtype=torch.float32, device=device)
    Z_t = torch.as_tensor(ZN, dtype=torch.float32, device=device)
    n_starts = T - horizon

    usid_model.eval()
    out = {}
    with torch.no_grad(), usid_model.natural_model.A_param.cache(), usid_model.A22_param.cache():
        P = usid_model.compute_photocurrents(u_tensor).squeeze(0)  # (T, k)
        x1 = Xf_t[:n_starts, :n1].clone()
        x2 = Xf_t[:n_starts, n1:].clone()
        start = torch.arange(n_starts, device=device)
        for j in range(horizon + 1):
            zhat = x1 @ Cz1_t.T + x2 @ Cz2_t.T          # decoded Z at depth j
            out[f"h{j}"] = _vw_r2(Z_t[start + j], zhat)
            if j < horizon:
                # self-fed y = C1·x1 + C2·x2 (the predictor free-run, forced=False)
                yhat = x1 @ usid_model.natural_model.Cy1.T + x2 @ usid_model.Cy2.T
                x1, x2, _ = usid_model.step(x1, x2, yhat, P[start + j])

    pred_h = 0
    for j in range(horizon + 1):
        r = out[f"h{j}"]
        if r == r and r > 0:  # r == r filters nan
            pred_h = j
        else:
            break
    out["predictable_horizon"] = int(pred_h)
    return out


def _forced_freerun_sse(usid_model, Xf_t, Y_t, P, starts, horizon, n1):
    """Self-fed observer-off forced-path free-run: SSE of output vs YN over depths
    j=0..horizon-1, with total output variance and sample count for a variance-weighted
    R2. From each filtered seed Xf[t0] the model self-feeds its own y, so opto reaches
    the output ONLY through B2' (via precomputed photocurrents P); the observer K2 acts
    on the self-fed innovation and is inert w.r.t. opto. Linear in B2'. Single source of
    the free-run rollout, shared by the B2' pin (grad on B2') and the multistart
    selection metric (eval), so the two cannot diverge.
    """
    x1 = Xf_t[starts, :n1]
    x2 = Xf_t[starts, n1:]
    sse = x1.new_zeros(())
    ytrue_pool = []
    for j in range(horizon):
        yhat = x1 @ usid_model.natural_model.Cy1.T + x2 @ usid_model.Cy2.T
        ytrue = Y_t[starts + j]
        sse = sse + ((yhat - ytrue) ** 2).sum()
        ytrue_pool.append(ytrue)
        x1, x2, _ = usid_model.step(x1, x2, yhat, P[starts + j])
    ytrue_pool = torch.cat(ytrue_pool, dim=0)
    ss_tot = ((ytrue_pool - ytrue_pool.mean(dim=0)) ** 2).sum()
    return sse, ss_tot, ytrue_pool.numel()


def _pin_bprime(usid_model, Xhat, YN, u_tensor, device, horizon, *, max_iter=300,
                train_frac=0.8):
    """Pin the opsin/forced input map B2' by a convex fit to the predictor free-run.

    The controller plans against the OBSERVER free-run (self-fed y = C·x) — the
    ``forced=False`` rollout in ``USIDModel.predict`` / ``NLMPC._rollout_state`` —
    where the unnatural update is x2[t+1] = (A22 + K2·Cy2)·x2 + B2'·photocurrent.
    Opto enters ONLY through B2'; the (possibly masked) observer K2 acts on the
    self-fed innovation Cy2·x2 and is therefore INERT w.r.t. opto in free-run.
    Because that free-run output is LINEAR in B2', fitting B2' to reproduce the true
    output YN over pooled rollout depths j=1..horizon is a CONVEX least-squares
    problem with a unique optimum. This identifies the input/forced path that the
    joint teacher-forced objective leaves Kalman-masked (the strong K·y_real feedback
    explains the opto response regardless of B2'), WITHOUT retraining any other
    matrix. B1' is pinned to 0 (opto kept out of the *direct* x1 input path; it can
    still reach x1 indirectly via A12·x2); all other params are frozen.

    Mirrors ``_observer_rollout_pool``'s rollout (reusing ``usid_model.step`` as the
    single source of the recurrence) but keeps grad on B2' and scores the free-run
    OUTPUT against YN. Seeds are the filtered states ``Xhat`` (held fixed during the
    fit; their weak B2'-dependence is a benign transient the horizon pooling averages
    over — generalization is checked on a held-out start split).

    Returns a diagnostic dict (``||ΔB2'||``, free-run output R²(Y) train/held-out).
    """
    n1, n2 = usid_model.n1, usid_model.n2
    T = Xhat.shape[0]
    if T <= horizon + 1:
        raise ValueError(
            f"Sequence too short to pin B': T={T}, horizon={horizon}. Need T > horizon + 1."
        )

    usid_model.eval()
    # Freeze everything; pin B1'=0; leave ONLY B2' trainable.
    for p in usid_model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        usid_model.B1_prime.zero_()
    usid_model.B2_prime.requires_grad_(True)
    B2_old = usid_model.B2_prime.detach().clone()

    Xf_t = torch.as_tensor(Xhat, dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(YN, dtype=torch.float32, device=device)
    # Photocurrents are fixed (opsin frozen) — precompute once, no grad.
    with torch.no_grad():
        P = usid_model.compute_photocurrents(u_tensor).squeeze(0)  # (T, k)

    n_starts = T - horizon
    n_train = max(1, int(round(train_frac * n_starts)))
    train_starts = torch.arange(n_train, device=device)
    val_starts = torch.arange(n_train, n_starts, device=device)
    if val_starts.numel() == 0:
        log.warning(
            "[pin_bprime] held-out start split is empty (n_starts=%d, train_frac=%g); "
            "held-out R2 will be nan.", n_starts, train_frac,
        )

    def free_run_sse(starts):
        # Single source of the free-run rollout (shared with the multistart selector
        # eval via _forced_freerun_sse) so the pin objective and the selection metric
        # cannot diverge. Keeps grad on B2' here (B2_prime.requires_grad_(True) above).
        return _forced_freerun_sse(usid_model, Xf_t, Y_t, P, starts, horizon, n1)

    opt = torch.optim.LBFGS(
        [usid_model.B2_prime], max_iter=max_iter, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        sse, _ss_tot, n = free_run_sse(train_starts)
        loss = sse / n
        loss.backward()
        return loss

    opt.step(closure)
    usid_model.B2_prime.requires_grad_(False)
    # Terminal freeze: usid_model is only READ after this (SubspaceLDS extraction,
    # Cz fit, Yhat) and never trained further, so leaving all params frozen is intended.

    with torch.no_grad():
        def r2(starts):
            if starts.numel() == 0:
                return float("nan")
            sse, ss_tot, _ = free_run_sse(starts)
            return float(1.0 - sse / ss_tot) if ss_tot > 0 else float("nan")
        diag = {
            "pin_bprime": True,
            "dB2_norm": float((usid_model.B2_prime - B2_old).norm()),
            "B2_old_norm": float(B2_old.norm()),
            "B2_new_norm": float(usid_model.B2_prime.norm()),
            "freerun_r2y_train": r2(train_starts),
            "freerun_r2y_heldout": r2(val_starts),
            "horizon": int(horizon),
        }
    log.info(
        f"[pin_bprime] ||dB2'||={diag['dB2_norm']:.4g} "
        f"(|B2'|: {diag['B2_old_norm']:.4g} -> {diag['B2_new_norm']:.4g}); "
        f"free-run R2(Y) train={diag['freerun_r2y_train']:.4f} "
        f"heldout={diag['freerun_r2y_heldout']:.4f} @h{horizon}"
    )
    return diag


def _freerun_r2y_heldout(usid_model, Xhat, YN, u_tensor, device, horizon, *, train_frac=0.8):
    """Held-out observer-off forced-path R2(Y) of the model AS-IS (no pin/optimization).

    Same self-fed free-run as ``_pin_bprime.free_run_sse`` (opto enters only through
    B2'; the self-fed observer is inert w.r.t. opto), evaluated on the held-out start
    split with the SAME train_frac so the split matches the pin's. Used as the
    multistart selection metric on the pin_bprime=False path: it ranks restarts by how
    well their forced path predicts held-out output — the control-relevant quantity the
    Kalman-masked validation loss is blind to. Returns nan if too short / empty split.
    """
    n1 = usid_model.n1
    T = Xhat.shape[0]
    if T <= horizon + 1:
        return float("nan")
    usid_model.eval()
    Xf_t = torch.as_tensor(Xhat, dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(YN, dtype=torch.float32, device=device)
    with torch.no_grad():
        P = usid_model.compute_photocurrents(u_tensor).squeeze(0)  # (T, k)
        n_starts = T - horizon
        n_train = max(1, int(round(train_frac * n_starts)))
        val_starts = torch.arange(n_train, n_starts, device=device)
        if val_starts.numel() == 0:
            return float("nan")
        sse, ss_tot, _ = _forced_freerun_sse(
            usid_model, Xf_t, Y_t, P, val_starts, horizon, n1)
        return float(1.0 - sse / ss_tot) if ss_tot > 0 else float("nan")


def _fit_behavioral_decoder(
    Xhat, ZN, n1, n2, *,
    usid_model=None, u_tensor=None, device=None,
    cz_fit_mode="teacher_forced", cz_observer_horizon=20, alpha=1.0,
):
    """Fit linear decoder from states to behavioral output Z.

    Two modes (see config/cold_usid.yaml usid_hyperparams.cz_fit_mode):

    * ``teacher_forced`` — Ridge-fit Cz on the 1-step teacher-forced (filtered)
      state ``Xhat``. This is the state the model is *trained/validated* against,
      but NOT the one the controller plans against. Legacy behaviour.

    * ``observer_rollout`` (default) — Ridge-fit Cz on the OBSERVER free-run
      state pooled over rollout depths ``j = 1..cz_observer_horizon`` (see
      ``_observer_rollout_pool``). This is the state the NLMPC/iLQR controllers
      actually visit during planning, so the decoder reads Z out of the right
      trajectory. Offline-validated to lift observer-form held-out R²(Z) at the
      control horizon (≈ −0.04 → +0.16 @h20) on the cached usid_stim_rec chr2
      cells.

    Uses fit_intercept=False because ZN is already zero-mean (normalized by
    model_fit.py). This ensures x2z(X) = X @ Cz.T with no bias term, matching
    the controller's expectation.
    """
    if cz_fit_mode == "teacher_forced":
        X_fit, Z_fit = Xhat, ZN
    elif cz_fit_mode == "observer_rollout":
        if usid_model is None or u_tensor is None or device is None:
            raise ValueError(
                "cz_fit_mode='observer_rollout' requires usid_model, u_tensor, "
                "and device."
            )
        # Xhat is the teacher-forced state trajectory (the controller's running
        # observer estimate); reused as the free-run rollout seeds.
        X_pool, Z_idx = _observer_rollout_pool(
            usid_model, Xhat, u_tensor, device, cz_observer_horizon
        )
        X_fit, Z_fit = X_pool, ZN[Z_idx]
    else:
        raise ValueError(
            f"Unknown cz_fit_mode={cz_fit_mode!r}; expected 'teacher_forced' "
            f"or 'observer_rollout'."
        )

    decoder = Ridge(alpha=alpha, fit_intercept=False)
    decoder.fit(X_fit, Z_fit)

    Cz_combined = decoder.coef_  # (nz, n1+n2)
    Cz1 = Cz_combined[:, :n1]
    Cz2 = Cz_combined[:, n1:]

    return Cz1, Cz2



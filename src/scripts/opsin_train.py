"""Train COLD's own opsin NN against the calibrated UnitlessChr2FourState plant.

Replaces the prior behaviour of reusing USID's synthetic-tagged opsin checkpoint.
The NN here mimics COLD's calibrated Bansal-derived plant *directly* via a
single-neuron Brian2 simulation — no fSNN, no transmittance projection. Inputs
are lognormal light traces drawn from the same distribution sys_id_expts uses
at deployment (``typical_Irr0_mWmm2``, ``u_max_mWmm2``), and targets are the
biophysical Iopto produced by ``UnitlessChr2FourState(g0, E)`` with
``g0, E`` read from the calibration JSON and ``V`` clamped to the calibrated
``v_median``.

Output: ``$USID_REPO_ROOT/experiments/opsin_results_external/cold/best_model.pth``
with provenance tag ``'external'`` and ``caller_tag='cold'``. The sentinel
this rule writes is consumed by COLD's ``model_fit`` rule so the cold-trained
checkpoint is the one used downstream — see ``src/model_fit/usid.py``.

``from __future__ import annotations`` is intentionally omitted: Snakemake's
``script:`` directive prepends a header above the module-top, which would
move the ``__future__`` import past line 1 and cause a SyntaxError.
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import brian2.only as b2
import numpy as np
from brian2 import ms
from joblib import Parallel, delayed

from src.opsins import UnitlessChr2FourState

# USID imports — USID_REPO_ROOT is set on PYTHONPATH by smk-submit-slurm.sbatch.
from scripts.opsin_train_external import train_opsin_external
from synthetic_data.synthetic_opsin_data import SynthOptoStimDataGen


log = logging.getLogger(__name__)


def _v_clamp_dummy_neuron_eqs(v_clamp: float) -> str:
    """Dimensionless v-clamped dummy neuron equations matching the form
    used by ``calibrate_opsin_plant._dimensionless_dummy_neuron_model`` and
    by USID's synthetic-data path. Required so cleo's transmittance / opsin
    injection passes dimension checks on the bare numeric ``v``/``Iopto``."""
    return (
        f"v = {v_clamp} : 1\n"
        f"            Iopto : 1\n"
        f"            x = 0*um : meter\n"
        f"            y = 0*um : meter\n"
        f"            z = 0*um : meter"
    )


def _generate_one_sequence(
    seq_idx: int,
    *,
    g0: float,
    E: float,
    v_clamp: float,
    mu: float,
    sigma: float,
    u_max_mWmm2: float,
    time_steps: int,
    model_period_ms: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single lognormal trace + photocurrent pair under a fresh
    Brian2 scope. Keeps loky workers isolated."""
    rng = np.random.default_rng(seed + seq_idx)
    trace = rng.lognormal(mu, sigma, size=time_steps).clip(0, u_max_mWmm2)

    b2.start_scope()
    opsin = UnitlessChr2FourState(g0=g0, E=E)
    data_gen = SynthOptoStimDataGen(
        opto_stim_fn=SynthOptoStimDataGen.make_array_stim_fn(
            trace, dt=model_period_ms * ms),
        opsin_model=opsin,
        model_period_ms=model_period_ms,
        dummy_neuron_model=_v_clamp_dummy_neuron_eqs(v_clamp),
    )
    duration_ms = time_steps * model_period_ms
    photocurrent_raw, light_raw = data_gen.run_simulation(duration_ms)
    p = photocurrent_raw[1]
    l = light_raw[1]
    if len(p) < time_steps:
        p = np.pad(p, (0, time_steps - len(p)), mode="edge")
    if len(l) < time_steps:
        l = np.pad(l, (0, time_steps - len(l)), mode="edge")
    return l[:time_steps].astype(np.float32), p[:time_steps].astype(np.float32)


def generate_cold_opsin_pool(
    *,
    g0: float,
    E: float,
    v_clamp: float,
    typical_Irr0_mWmm2: float,
    u_max_mWmm2: float,
    n_sequences: int,
    time_steps: int,
    model_period_ms: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drive N independent single-neuron sims; stack to (N, T, 1) arrays."""
    mu = float(np.log(typical_Irr0_mWmm2))
    sigma = float(np.log(u_max_mWmm2) / 3.0)

    if verbose:
        log.info(
            f"Generating {n_sequences} COLD opsin sequences (single-neuron, no fSNN).")
        log.info(
            f"  Calibration: g0={g0:.4g}, E={E:.4f}, v_clamp={v_clamp:.4f}")
        log.info(
            f"  Light dist:  lognormal(mu={mu:.4f}, sigma={sigma:.4f}) "
            f"clipped to [0, {u_max_mWmm2}] mW/mm²")
        log.info(
            f"  Duration:    {time_steps} steps × {model_period_ms} ms = "
            f"{time_steps * model_period_ms:.0f} ms per sequence")

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    n_jobs = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    if verbose:
        log.info(f"  Parallel workers: {n_jobs}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_generate_one_sequence)(
            i,
            g0=g0, E=E, v_clamp=v_clamp,
            mu=mu, sigma=sigma, u_max_mWmm2=u_max_mWmm2,
            time_steps=time_steps,
            model_period_ms=model_period_ms,
            seed=seed,
        )
        for i in range(n_sequences)
    )

    light_arr = np.stack([r[0] for r in results]).reshape(n_sequences, time_steps, 1)
    photo_arr = np.stack([r[1] for r in results]).reshape(n_sequences, time_steps, 1)
    return light_arr, photo_arr


def main(snakemake) -> None:
    config = snakemake.config

    calib_path = Path(snakemake.input.calibration)
    calib = json.loads(calib_path.read_text())
    g0 = float(calib["g0"])
    E = float(calib["E"])
    v_clamp = float(calib["v_median"])

    train_cfg = config.get("opsin_train_cold") or {}
    if not train_cfg:
        raise ValueError(
            "Missing 'opsin_train_cold' config block. Add it to the params "
            "YAML (see config/cold_usid.yaml or testconfig_params.yaml).")

    n_sequences = int(train_cfg["n_sequences"])
    time_steps = int(train_cfg["time_steps"])
    sequence_length = int(train_cfg.get("sequence_length", time_steps))
    seed = int(train_cfg["seed"])
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    lr = float(train_cfg["lr"])
    hidden_size = int(train_cfg.get("hidden_size", 32))
    num_layers = int(train_cfg.get("num_layers", 2))
    scheduler_patience = int(train_cfg.get("scheduler_patience", 200))
    scheduler_factor = float(train_cfg.get("scheduler_factor", 0.8))
    scheduler_min_lr = float(train_cfg.get("scheduler_min_lr", 1e-6))

    typical_Irr0_mWmm2 = float(config["cleo_prep"]["typical_Irr0_mWmm2"])
    u_max_mWmm2 = float(config["expts"]["u_max_mWmm2"])
    model_period_ms = float(
        config["prep_fsnn"]["dt_ms"] * config["cleo_prep"]["subsample_factor"])

    light, photocurrent = generate_cold_opsin_pool(
        g0=g0, E=E, v_clamp=v_clamp,
        typical_Irr0_mWmm2=typical_Irr0_mWmm2,
        u_max_mWmm2=u_max_mWmm2,
        n_sequences=n_sequences,
        time_steps=time_steps,
        model_period_ms=model_period_ms,
        seed=seed,
    )

    usid_repo = os.environ.get("USID_REPO_ROOT")
    if not usid_repo:
        raise EnvironmentError(
            "USID_REPO_ROOT env var must be set so the cold-trained "
            "checkpoint lands where load_trained_opsin can find it. "
            "The COLD sbatch script sets this automatically.")
    # Timestamped subdir under cold/ so ``load_trained_opsin`` can use its
    # ``_find_latest_opsin_checkpoint`` discovery (which iterates subdirs of
    # the given parent and lex-sorts by name to pick the newest). Matches the
    # convention USID synthetic opsin training uses under
    # ``experiments/opsin_results/<timestamp>/``.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(usid_repo) / "experiments" / "opsin_results_external" / "cold" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    best_path = train_opsin_external(
        light=light,
        photocurrent=photocurrent,
        input_scale_ceiling=u_max_mWmm2,
        caller_tag="cold",
        save_dir=save_dir,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_size=hidden_size,
        num_layers=num_layers,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        scheduler_min_lr=scheduler_min_lr,
        val_split=0.15,
        test_split=0.10,
        seed=seed,
        plot_data_eda=True,
    )

    sentinel_path = Path(snakemake.output.sentinel)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(str(best_path))
    log.info(f"[opsin_train_cold] Best checkpoint: {best_path}")
    log.info(f"[opsin_train_cold] Wrote sentinel: {sentinel_path}")


if __name__ == "__main__":
    from snakemake.script import snakemake  # type: ignore

    from src.helpers import setup_logging  # type: ignore

    setup_logging(snakemake.config, snakemake.log[0])
    main(snakemake)

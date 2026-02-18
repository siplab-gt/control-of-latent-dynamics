from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from npeet_plus import mi, midc
from scipy.stats import bootstrap

import src.plot_style as psty
from src.helpers import save_pkl_gz, setup_debugger


@dataclass
class CMIEstimate:
    ep: str
    n_samples: int
    cmi: float
    p_value: float
    q_value: float
    ci_low: float
    ci_high: float
    null_dist: np.ndarray
    boot_dist: np.ndarray


@dataclass
class CMIResult:
    setting: str
    task: str
    analysis: str
    metric_name: str
    conditioning_vars: list[str]
    n_shuffles: int
    n_bootstraps: int
    random_seed: int
    ep_results: list[CMIEstimate] = field(default_factory=list)

SEED = 18810929

def _group_ids_from_z(z_df):
    if z_df.shape[1] == 1:
        return pd.factorize(z_df.iloc[:, 0], sort=True)[0]
    idx = pd.MultiIndex.from_frame(z_df)
    return pd.factorize(idx, sort=True)[0]


def _shuffle_within_groups(y, group_ids, rng):
    shuffled = y.copy()
    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        rng.shuffle(shuffled[idx])
    return shuffled


def _estimate_cmi(x, y, z, discrete):
    if discrete:
        return midc(x, y, z)
    else:
        return mi(x, y, z)


def _bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranked
    qvals[order] = np.minimum.accumulate(qvals[order][::-1])[::-1]
    return np.clip(qvals, 0, 1)


def _compute_bootstrap_ci(x, y, z, discrete, n_boot, rng):
    def _stat(x_s, y_s, z_s):
        return _estimate_cmi(x_s, y_s, z_s, discrete)

    res = bootstrap(
        (x, y, z),
        _stat,
        n_resamples=n_boot,
        method="BCa",
        random_state=rng,
    )
    try:
        boot_dist = np.asarray(res.bootstrap_distribution)
    except AttributeError:
        boot_dist = np.array([])
    return res.confidence_interval.low, res.confidence_interval.high, boot_dist


def _prepare_arrays(df, x_col, y_col, z_cols):
    data = df[[x_col, y_col] + z_cols].dropna()
    if data.empty:
        return None, None

    x_is_bool = pd.api.types.is_bool_dtype(data[x_col])
    x = data[x_col]
    y = data[y_col]
    z_df = data[z_cols].copy()

    if x_is_bool:
        # Encode boolean as float for midc compatibility if needed, or integers
        x = x.astype(int)

    # Convert to numpy
    x = x.to_numpy()
    y = y.to_numpy()
    z = z_df.to_numpy()

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    return x, y, z, data, x_is_bool


def main(cfg, results_csv, outfile, setting, task, analysis):
    psty.style_plots()
    df = pd.read_csv(results_csv) if results_csv else None

    expl_params = cfg["star_sweep_params"].copy()
    if setting == "optimal_stim_rec":
        expl_params |= cfg["star_sweep_params_measurement_only"]

    if setting == "sufficient_stim":
        metric = "Q_track_Xref_X_fsnn"
        key_metric_name = "k_suff"
    else:
        metric = "Q_track_Zref_Z"
        key_metric_name = "optimal_alpha_stim"

    ref_condition = cfg["reference_params"]

    if analysis not in {"key", "control"}:
        raise ValueError(f"Unknown analysis: {analysis}")

    n_shuffles = 1000
    n_bootstraps = 1000
    rng = np.random.default_rng(SEED)

    if analysis == "key":
        metric_name = key_metric_name
        conditioning_vars = ["seed"]
    else:
        metric_name = metric
        conditioning_vars = ["seed", "k"]

    result = CMIResult(
        setting=setting,
        task=task,
        analysis=analysis,
        metric_name=metric_name,
        conditioning_vars=conditioning_vars,
        n_shuffles=n_shuffles,
        n_bootstraps=n_bootstraps,
        random_seed=SEED,
    )

    for ep in expl_params:
        if analysis == "key":
            subset = df.query("ep == @ep and key_metric_name == @key_metric_name")
            x_col = "ep_value"
            y_col = "key_metric_value"
            z_cols = ["seed"]
        else:
            conds2keep = ref_condition.copy()
            del conds2keep[ep], conds2keep["latency"]
            query_str = " and ".join(
                [f"({k} == {v!r} or {k}.isnull())" for k, v in conds2keep.items()]
            )
            query_str += ' and (ctrl_latency == "MPC")'
            subset = df.query(query_str)
            x_col = ep
            y_col = metric
            z_cols = ["seed", "k"]

        if subset.empty:
            continue

        prep_res = _prepare_arrays(subset, x_col, y_col, z_cols)
        if prep_res[0] is None:
            continue
        x, y, z, data, is_discrete = prep_res

        cmi_val = _estimate_cmi(x, y, z, is_discrete)

        group_ids = _group_ids_from_z(data[z_cols])
        null_dist = np.zeros(n_shuffles)
        for i in range(n_shuffles):
            y_shuffled = _shuffle_within_groups(y, group_ids, rng)
            null_dist[i] = _estimate_cmi(x, y_shuffled, z, is_discrete)

        p_value = (np.sum(null_dist >= cmi_val) + 1) / (n_shuffles + 1)

        ci_low, ci_high, boot_dist = _compute_bootstrap_ci(
            x, y, z, is_discrete, n_bootstraps, rng
        )

        result.ep_results.append(
            CMIEstimate(
                ep=str(ep),
                n_samples=len(x),
                cmi=float(cmi_val),
                p_value=float(p_value),
                q_value=float("nan"),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                null_dist=null_dist,
                boot_dist=boot_dist,
            )
        )

    pvals = [entry.p_value for entry in result.ep_results]
    if pvals:
        qvals = _bh_fdr(pvals)
        for entry, qval in zip(result.ep_results, qvals, strict=True):
            entry.q_value = float(qval)

    save_pkl_gz(result, outfile)
    print(f"Saved CMI results to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(
        snakemake.config,
        snakemake.input[0],
        snakemake.output[0],
        setting=snakemake.wildcards.setting,
        task=snakemake.wildcards.task,
        analysis=snakemake.wildcards.analysis,
    )


if __name__ == "__main__" and "snakemake" not in locals():
    import argparse
    from pathlib import Path
    import snakemake

    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Path to results CSV file")
    args = parser.parse_args()

    results_file = args.results.resolve()
    filename = results_file.name

    if "key_metrics" in filename:
        analysis = "key"
    else:
        analysis = "control"

    stem = results_file.stem
    # strip _key_metrics if present
    if analysis == "key":
        stem = stem.replace("_key_metrics", "")

    if stem.startswith("sufficient_stim_"):
        setting = "sufficient_stim"
        task = stem[len("sufficient_stim_") :]
    elif stem.startswith("optimal_stim_rec_"):
        setting = "optimal_stim_rec"
        task = stem[len("optimal_stim_rec_") :]
    else:
        raise ValueError(
            f"Could not infer setting from filename '{filename}'. "
            "Expected to start with 'sufficient_stim' or 'optimal_stim_rec'."
        )

    outfile = results_file.parent / f"{setting}_{task}_cmi_{analysis}.pkl.gz"

    print(f"Inferred: setting={setting}, task={task}, analysis={analysis}")
    print(f"Output: {outfile}")

    # Load config
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        # try relative to script
        config_path = Path(__file__).parent.parent.parent / "config/config.yaml"

    if not config_path.exists():
        raise FileNotFoundError("Could not find config/config.yaml")

    cfg = snakemake.common.configfile._load_configfile(str(config_path))

    main(
        cfg,
        str(results_file),
        str(outfile),
        setting=setting,
        task=task,
        analysis=analysis,
    )

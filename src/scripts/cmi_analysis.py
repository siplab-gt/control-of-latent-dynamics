from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from npeet_plus import mi, micd
from scipy.stats import bootstrap
from jaxtyping import Float

import src.plot_style as psty
from src.helpers import setup_debugger
from src.data_classes import BaseDataClass


@dataclass
class CMIEstimate(BaseDataClass):
    ep: str
    n_samples: int
    cmi: float
    p_value: float
    q_value: float
    ci_low: float
    ci_high: float
    null_dist: Float[np.ndarray, "*"]
    boot_dist: Float[np.ndarray, "*"]


@dataclass
class CMIResult(BaseDataClass):
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
        # shuffle via indices to handle numpy advanced indexing copy issues
        perm = rng.permutation(len(idx))
        shuffled[idx] = shuffled[idx][perm]
    return shuffled


def _estimate_cmi(x, y, z, discrete):
    k = min(3, len(x) - 1)
    if discrete:
        # flip since midc is broken
        res = micd(y, x, z, k=k)
    else:
        res = mi(x, y, z, k=k)
    assert np.isfinite(res), f"CMI must be finite; got {res}"
    return res


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
    def _stat(idx):
        return _estimate_cmi(x[idx], y[idx], z[idx], discrete)

    # data is a tuple of (indices,)
    data = (np.arange(len(x)),)

    res = bootstrap(
        data,
        _stat,
        n_resamples=n_boot,
        method="BCa",
        random_state=rng,
        paired=True,
    )

    # Use atleast_1d to handle both scalar and array returns from bootstrap
    ci_low = np.atleast_1d(res.confidence_interval.low)[0]
    ci_high = np.atleast_1d(res.confidence_interval.high)[0]

    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        print(
            "Warning: bootstrap confidence interval is not finite: "
            f"{res.confidence_interval}. Using percentile method."
        )
        res = bootstrap(
            data,
            _stat,
            n_resamples=n_boot,
            method="percentile",
            random_state=rng,
            paired=True,
        )
        ci_low = np.atleast_1d(res.confidence_interval.low)[0]
        ci_high = np.atleast_1d(res.confidence_interval.high)[0]

    try:
        boot_dist = np.asarray(res.bootstrap_distribution)
    except AttributeError:
        boot_dist = np.array([])
    return ci_low, ci_high, boot_dist


def _prepare_arrays(df: pd.DataFrame, x_col, y_col, z_cols):
    data = df[[x_col, y_col] + z_cols].dropna()
    if data.empty:
        return None, None

    x_vals = data[x_col]

    # Infer if x is discrete (boolean or string/categorical)
    # Check if we can convert to numeric (handles "10", "11" in object col)
    try:
        x_vals = pd.to_numeric(x_vals, errors="raise")
    except (ValueError, TypeError):
        pass

    inferred_type = pd.api.types.infer_dtype(x_vals, skipna=True)
    x_is_discrete = inferred_type in ("boolean", "string", "unicode")

    if x_is_discrete:
        if inferred_type in ("string", "unicode"):
            x_vals, _ = pd.factorize(x_vals)
        else:
            x_vals = x_vals.astype(int)

    y = data[y_col]
    z_df = data[z_cols].copy()

    # Convert to numpy
    x = np.asarray(x_vals)
    y = y.to_numpy()
    z = z_df.to_numpy()

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    return x, y, z, data, x_is_discrete


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
            query_str += ' and (ctrl_type == "LQMPC")'
            subset = df.query(query_str)
            x_col = ep
            y_col = metric
            z_cols = ["seed", "k"]

        if subset.empty:
            continue

        prep_res = _prepare_arrays(subset, x_col, y_col, z_cols)
        if prep_res[0] is None:
            continue

        # Explicit unpacking for linter clarity
        x, y, z, data, is_discrete = prep_res
        # Type hints / checks effectively
        if x is None or y is None or z is None:
            continue

        if len(x) <= 3:
            print(f"Skipping {ep}: too few samples ({len(x)})")
            continue

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
        assert ci_low <= cmi_val <= ci_high, f"{ep=}, {cmi_val=}, {ci_low=}, {ci_high=}"

        print(f"{ep=}, {cmi_val=}, {p_value=}, {ci_low=}, {ci_high=}")
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
            print(f"{entry.ep}: q = {entry.q_value}")

    result.save(outfile)
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

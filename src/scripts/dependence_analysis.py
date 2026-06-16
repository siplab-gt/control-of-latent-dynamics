import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from jaxtyping import Float

import src.plot_style as psty
from src.cmi_estimation import (
    statistical_validation_bootstrap,
    statistical_validation_permutation,
)
from src.data_classes import BaseDataClass
from src.helpers import preprocess_params, setup_debugger, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class CMIEstimate(BaseDataClass):
    ep: str
    n_samples: int
    cmi: float
    p_value: float
    q_value: float
    ci_low: float
    ci_high: float
    null_dist: Float[np.ndarray, "n_shuffles"] = field(repr=False)
    boot_dist: Float[np.ndarray, "n_bootstraps"] = field(repr=False)

    @property
    def summary_dict(self):
        return {
            "ep": self.ep,
            "n_samples": self.n_samples,
            "null_min": np.min(self.null_dist),
            "null_mean": np.mean(self.null_dist),
            "null_max": np.max(self.null_dist),
            "ci_low": self.ci_low,
            "cmi": self.cmi,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "q_value": self.q_value,
        }

    def print_summary(self):
        with np.printoptions(precision=3, suppress=True):
            print(pd.DataFrame([self.summary_dict]))


class NullCorrectedCMIEstimate(CMIEstimate):
    """CMI estimate with null distribution bias correction."""

    cmi_estimate: CMIEstimate

    def __init__(self, cmi_estimate: CMIEstimate):
        self.cmi_estimate = cmi_estimate

    def __getattr__(self, name):
        if name in ["cmi", "ci_low", "ci_high", "null_dist", "boot_dist"]:
            return getattr(self.cmi_estimate, name) - np.mean(
                self.cmi_estimate.null_dist
            )
        else:
            return getattr(self.cmi_estimate, name)


SEED = 18810929


def _group_ids_from_z(z_df):
    if z_df.shape[1] == 1:
        return pd.factorize(z_df.iloc[:, 0], sort=True)[0]
    idx = pd.MultiIndex.from_frame(z_df)
    return pd.factorize(idx, sort=True)[0]


def _prepare_arrays(df: pd.DataFrame, x_col, y_col, z_cols):
    data = df[[x_col, y_col] + z_cols]
    assert not data.empty

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

    x = np.asarray(x_vals)
    y = data[y_col].to_numpy()
    z = data[z_cols].to_numpy()

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    return x, y, z, data, x_is_discrete


def main(wildcards, cfg, base_params, results_csv, outfile):
    psty.style_plots()
    df = pd.read_csv(results_csv)

    setting = wildcards.setting
    task = wildcards.task
    analysis = wildcards.analysis
    ep = wildcards.ep

    expl_params = cfg["star_sweep_params"].copy() | cfg[f"star_sweep_params_{setting}"]

    ref_condition = cfg["reference_params"] | cfg[f"reference_params_{setting}"]

    if analysis not in {"key", "control"}:
        raise ValueError(f"Unknown analysis: {analysis}")

    rng = np.random.default_rng(SEED)

    if analysis == "key":
        metric_name = base_params.key_metrics[0]
        conditioning_vars = ["seed"]
    else:
        metric_name = base_params.ctrl_metric
        conditioning_vars = ["seed", "k"]

    n_shuffle_boot = base_params["n_shuffle_boot"]

    if analysis == "key":
        subset = df.query("ep == @ep")
        x_col = "ep_value"
        y_col = metric_name
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
        y_col = metric_name
        z_cols = ["seed", "k"]

    assert not subset.empty, f"No data for {ep}"
    assert len(subset[y_col].unique()) > 1, (
        f"y should vary for {ep}. {subset[y_col].unique()=}"
    )
    logger.info(f"Using {subset.shape[0]} samples for {ep} {analysis}")

    x, y, z, data, x_is_cat = _prepare_arrays(subset, x_col, y_col, z_cols)

    assert len(x) > 3, f"Too few samples for {ep}. {len(x)=}"

    group_ids = _group_ids_from_z(data[z_cols])
    for gid in np.unique(group_ids):
        x_gid = x[group_ids == gid]
        assert np.var(x_gid) > 0, f"x should vary within groups for {ep}. {x_gid=}"
        y_gid = y[group_ids == gid]
        if metric_name not in [
            "optimal_alpha_stim",
            "k_knee",
        ]:  # likely to repeat exact values
            assert np.var(y_gid) > 0, f"y should vary within groups for {ep}. {y_gid=}"

    # we will treat categorical variables as discrete unordered but
    # other numerical X and Z (like k or n trials) as discrete ordered
    cat_XZ = []
    cat_Z = []
    var_types = ""
    if x_is_cat:
        cat_XZ.append(0)
        var_types += "u"  # unordered discrete
    else:
        var_types += "o"  # ordered discrete

    var_types += "c"  # y is continuous

    for i, z_var in enumerate(z_cols):
        if z_var in ["seed"]:
            cat_Z.append(i)
            cat_XZ.append(1 + i)
            var_types += "u"  # unordered discrete
        elif z_var in ["k"]:
            var_types += "o"  # ordered discrete
        else:
            raise ValueError(f"Unknown Z variable: {z_var}")

    logger.info(f"Using {wildcards.estimator_type} for CMI estimation for EP {ep}")

    cluster_ids = _group_ids_from_z(data[["seed"]])

    cv_groups = _group_ids_from_z(data[["seed"]])
    common_kwargs = dict(
        x=x,
        y=y,
        z=z,
        group_ids=group_ids,
        estimator_type=wildcards.estimator_type,
        cat_XZ=cat_XZ,
        cat_Z=cat_Z,
        var_types=var_types,
        rng=rng,
        cv_groups=cv_groups,
    )

    p_value, cmi_val, null_dist = statistical_validation_permutation(
        **common_kwargs, n_permutations=n_shuffle_boot
    )

    if ep != "model_task_inputs":
        assert np.min(null_dist) != np.max(null_dist), (
            f"Null distribution shouldn't be constant. {ep=}. {null_dist[0]=}"
        )

    ci_low, ci_high, boot_dist = statistical_validation_bootstrap(
        **common_kwargs, n_boot=n_shuffle_boot, cluster_ids=cluster_ids
    )
    assert ci_low <= cmi_val <= ci_high, f"{ep=}, {cmi_val=}, {ci_low=}, {ci_high=}"

    result = CMIEstimate(
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
    result.print_summary()

    result.save(outfile)
    print(f"Saved CMI results to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))
    log_fh = setup_logging(snakemake.config, snakemake.log[0])

    main(
        snakemake.wildcards,
        snakemake.config,
        preprocess_params(snakemake.params),
        snakemake.input[0],
        snakemake.output[0],
    )


if __name__ == "__main__" and "snakemake" not in locals():
    import argparse
    from pathlib import Path

    import snakemake

    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Path to results CSV file")
    parser.add_argument("--ep", type=str, required=True, help="Explanatory variable")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    args = parser.parse_args()

    setup_debugger(args.debug)

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
        preprocess_params(cfg).dependence_analysis,
        str(results_file),
        str(outfile),
        setting=setting,
        task=task,
        analysis=analysis,
        ep=args.ep,
    )

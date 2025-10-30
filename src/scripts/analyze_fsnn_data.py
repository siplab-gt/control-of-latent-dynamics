import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.helpers import load_pkl_gz


def analyze_fsnn_data(cond_params, infile, outfile):
    data = load_pkl_gz(infile)

    # exclude first point since network needs a step to update y
    y_target = data["y_mon"]["ytilde"][1:]
    y_fsnn = data["y_mon"]["y_"][1:]
    z_target = data["z_mon"]["fout"][1:]
    z_fsnn = data["z_mon"]["z"][1:]

    metrics = {}

    for y_true, y_pred, z_true, z_pred, label in [
        (y_target, y_fsnn, z_target, z_fsnn, "target"),
        (y_fsnn, y_target, z_fsnn, z_target, "fsnn"),
    ]:
        # without force_finite, flat true series should yield -Inf
        # this would be the case for the 3rd reaching output
        # we will ignore these when computing summary statistics
        factor_r2s = r2_score(
            y_true, y_pred, multioutput="raw_values", force_finite=False
        )
        output_r2s = r2_score(
            z_true, z_pred, multioutput="raw_values", force_finite=False
        )

        metrics |= {
            f"R2_fsnn_{label}_factors_mean": np.mean(
                factor_r2s, where=np.isfinite(factor_r2s)
            ),
            f"R2_fsnn_{label}_factors_std": np.std(
                factor_r2s, where=np.isfinite(factor_r2s)
            ),
            f"R2_fsnn_{label}_factors_median": np.median(
                factor_r2s[np.isfinite(factor_r2s)]
            ),
            f"R2_fsnn_{label}_output_mean": np.mean(
                output_r2s, where=np.isfinite(output_r2s)
            ),
            f"R2_fsnn_{label}_output_std": np.std(
                output_r2s, where=np.isfinite(output_r2s)
            ),
            f"R2_fsnn_{label}_output_median": np.median(
                output_r2s[np.isfinite(output_r2s)]
            ),
        }
        for i, r2 in enumerate(factor_r2s):
            metrics[f"R2_fsnn_{label}_factor_{i}"] = r2
        for i, r2 in enumerate(output_r2s):
            metrics[f"R2_fsnn_{label}_output_{i}"] = r2

    metrics |= {
        "RMSE_fsnn_factors": np.sqrt(np.mean((y_target - y_fsnn) ** 2)),
        "RMSE_fsnn_output": np.sqrt(np.mean((z_target - z_fsnn) ** 2)),
        "MSE_fsnn_factors": np.mean((y_target - y_fsnn) ** 2),
        "MSE_fsnn_output": np.mean((z_target - z_fsnn) ** 2),
        "n_factors": y_target.shape[1],
    }
    print(metrics)
    print(f"Saving metrics values to {outfile}")
    pd.DataFrame([metrics]).to_csv(outfile, index=False)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    analyze_fsnn_data(snakemake.wildcards, snakemake.input[0], snakemake.output[0])

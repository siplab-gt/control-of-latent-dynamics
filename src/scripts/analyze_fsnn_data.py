import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from helpers import load_pkl_gz


def analyze_fsnn_data(cond_params, infile, outfile):
    data = load_pkl_gz(infile)

    y_target = data["y_mon"]["ytilde"]
    y_fsnn = data["y_mon"]["y_"]
    z_target = data["z_mon"]["fout"]
    z_fsnn = data["z_mon"]["z"]

    factor_r2s = r2_score(y_target, y_fsnn, multioutput="raw_values")
    output_r2s = r2_score(z_target, z_fsnn, multioutput="raw_values")
    metrics = {
        "R2_fsnn_factors_mean": np.mean(factor_r2s),
        "R2_fsnn_factors_std": np.std(factor_r2s),
        "R2_fsnn_factors_median": np.median(factor_r2s),
        "R2_fsnn_output_mean": np.mean(output_r2s),
        "R2_fsnn_output_std": np.std(output_r2s),
        "R2_fsnn_output_median": np.median(output_r2s),
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

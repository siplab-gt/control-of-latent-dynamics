import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.scripts.model_fit import ModelFitPredData
from src.scripts.sys_id_expts import ExptData


def analyze_model_fit_data(
    cond_params, model_fit_pred_data: ModelFitPredData, sys_id_data: ExptData, outfile
):
    # T = sys_id_data.i_trial_boundaries[int(cond_params.n_trials_sys_id) - 1, -1]
    T = len(sys_id_data.t_ms)
    X = sys_id_data.X[:T]
    Y = sys_id_data.Y[:T]
    Z = sys_id_data.Z[:T]
    # save R2 for model fit quality
    r2s = {
        "R2_model_fit_X_Xhat_fsnn": r2_score(X, model_fit_pred_data.Xhat_lds2fsnn),
        "R2_model_fit_Z_Zhat": r2_score(
            Z, model_fit_pred_data.Zhat, multioutput="variance_weighted"
        ),
        # sanity check: should be 1
        "R2_model_fit_Z_X": LinearRegression().fit(X, Z).score(X, Z),
        "R2_model_fit_Y_Yhat": r2_score(Y, model_fit_pred_data.Yhat)
        if Y.shape[1] > 0
        else np.nan,
    }
    print(r2s)
    print(f"Saving R2 values to {outfile}")
    pd.DataFrame([r2s]).to_csv(outfile, index=False)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    model_fit_pred_data = ModelFitPredData.load(snakemake.input[0])
    sys_id_data = ExptData.load(snakemake.input[1])

    analyze_model_fit_data(
        snakemake.wildcards, model_fit_pred_data, sys_id_data, snakemake.output[0]
    )

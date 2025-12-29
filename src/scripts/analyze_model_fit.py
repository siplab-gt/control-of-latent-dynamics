import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.scripts.model_fit import LDS, ModelFitPredData
from src.scripts.sys_id_expts import ExptData


def analyze_model_fit(
    cond_params,
    lds: LDS,
    model_fit_pred_data: ModelFitPredData,
    sys_id_data: ExptData,
    outfile,
):
    # cut data to the beginning of the n+1th trial
    try:
        T = sys_id_data.i_trial_boundaries[int(cond_params.n_trials_sys_id), 0]
    except IndexError:
        T = len(sys_id_data.X)
    X = sys_id_data.X[:T]
    Y = sys_id_data.Y[:T]
    Z = sys_id_data.Z[:T]
    Zhat_from_X = LinearRegression().fit(X, Z).predict(X)
    # save R2 for model fit quality
    metrics = {
        "R2_model_fit_X_Xhat_fsnn": r2_score(X, model_fit_pred_data.Xhat_lds2fsnn),
        "R2_model_fit_Z_Zhat": r2_score(
            Z, model_fit_pred_data.Zhat, multioutput="variance_weighted"
        ),
        # sanity check: should be 1
        # variance_weighted should account for the constant 3rd output for reaching.
        "R2_model_fit_Z_X": r2_score(Z, Zhat_from_X, multioutput="variance_weighted"),
        "R2_model_fit_Y_Yhat": r2_score(Y, model_fit_pred_data.Yhat)
        if Y.shape[1] > 0
        else np.nan,
    }
    metrics["largest_eigval"] = np.max(np.abs(np.linalg.eigvals(lds.A)))
    print(metrics)
    print(f"Saving R2 values to {outfile}")
    pd.DataFrame([metrics]).to_csv(outfile, index=False)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    lds = LDS.load(snakemake.input[0])
    model_fit_pred_data = ModelFitPredData.load(snakemake.input[1])
    sys_id_data = ExptData.load(snakemake.input[2])

    analyze_model_fit(
        snakemake.wildcards, lds, model_fit_pred_data, sys_id_data, snakemake.output[0]
    )

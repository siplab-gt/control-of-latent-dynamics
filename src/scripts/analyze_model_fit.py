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

    # Sanitize clip counts from the normalization pipeline. A non-zero
    # `n_sanitize_clipped_{Y,Z,U,X,Natural_YN}` column flags a pathological
    # row: ±1e6 clipping kicked in during `sanitize_normalized` after
    # float32 cast would have overflowed, which typically means v3 plant
    # divergence. The clipped channel's contribution is meaningless even
    # though the array is finite — summary aggregation should filter.
    clip_counts = getattr(lds, "sanitize_clip_counts", {}) or {}
    for key in ("Y", "Z", "U", "X", "Natural_YN"):
        metrics[f"n_sanitize_clipped_{key}"] = int(clip_counts.get(key, 0))

    # Horizon-resolved free-run R²(Z) (the load-bearing controllability metric) and the
    # pin-B' diagnostic, both stashed on lds.metadata by model_fit/usid.py. Absent for
    # non-USID fits (BRAID/known_factors) — NaN-filled then. R2_freerun_Z_h0 ≈
    # R2_model_fit_Z_Zhat is the correctness gate; R2_freerun_Z_h7/h10/h20 (70/100/200 ms;
    # 70 ms ≈ the chr2 opsin time constant) are what actually predict control; pin_dB2_norm
    # > 0 confirms the pin took effect. The metric computes every depth h0..cz_observer_horizon,
    # so adding/removing an `h` here is purely a column selection over the saved curve.
    meta = getattr(lds, "metadata", {}) or {}
    fr = meta.get("freerun_r2z", {}) or {}
    for h in (0, 1, 5, 7, 10, 20):
        metrics[f"R2_freerun_Z_h{h}"] = float(fr.get(f"h{h}", np.nan))
    metrics["freerun_Z_predictable_horizon"] = float(fr.get("predictable_horizon", np.nan))
    pin = meta.get("pin_bprime", {}) or {}
    metrics["pin_bprime"] = bool(pin.get("pin_bprime", False))
    metrics["pin_dB2_norm"] = float(pin.get("dB2_norm", np.nan))
    metrics["pin_freerun_r2y_heldout"] = float(pin.get("freerun_r2y_heldout", np.nan))

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

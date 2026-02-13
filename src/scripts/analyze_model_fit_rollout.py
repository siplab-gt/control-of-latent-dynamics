import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.helpers import str2bool
from src.scripts.model_fit import LDS, ModelFitPredData
from src.scripts.sys_id_expts import ExptData


def analyze_model_fit_rollout(
    cond_params,
    lds,
    sys_id_data: ExptData,
    outfile,
    max_horizon=None,
):
    """
    Calculate R² for each prediction horizon across entire dataset
    For horizon i: compare Xhat[i:n] (computed from X[0:n-i]) with Xtrue[i:n]

    Efficiently computes by rolling forward one step at a time, computing R² at each horizon.

    Args:
        cond_params: Condition parameters
        lds: LDS model with A, B, C matrices and normalization parameters
        model_fit_pred_data: Model fit prediction data (used for timing reference)
        sys_id_data: Experimental data with ground truth and trial boundaries
        outfile: Output CSV file path
        max_horizon: Maximum prediction horizon (defaults to first trial length)
    """
    print(f"Analyzing model fit rollout for condition: {cond_params}")

    # Determine data length (use same cutoff as in analyze_model_fit.py)
    try:
        T = sys_id_data.i_trial_boundaries[int(cond_params.n_trials_sys_id), 0]
    except (IndexError, AttributeError):
        T = len(sys_id_data.X)

    # Ground truth data
    X_true = sys_id_data.X[:T]
    Z_true = sys_id_data.Z[:T]
    Y_true = getattr(sys_id_data, "Y", None)
    if Y_true is not None:
        Y_true = Y_true[:T]

    # Prepare input matrices
    model_task_inputs = getattr(cond_params, "model_task_inputs", False)
    if str2bool(model_task_inputs):
        U = np.hstack([sys_id_data.Uopto[:T], sys_id_data.Utask[:T]])
    else:
        U = sys_id_data.Uopto[:T]

    # Normalize inputs using LDS parameters
    U_norm = (U - lds.u_mean) / lds.u_std

    # Determine max prediction horizon (samples in first trial)
    if max_horizon is None:
        if (
            hasattr(sys_id_data, "i_trial_boundaries")
            and sys_id_data.i_trial_boundaries is not None
        ):
            # Use first trial length
            first_start, first_end = sys_id_data.i_trial_boundaries[0]
            max_horizon = first_end - first_start - 1  # Can't predict from last point
        else:
            max_horizon = min(500, T - 1)  # Fallback
    # pad with zeros to simplify rollout past T
    U_norm = np.vstack([U_norm, np.zeros((max_horizon, U_norm.shape[1]))])

    print(f"Using max_horizon: {max_horizon}")
    print(f"Data length: {T}")

    # Normalize states for LDS computation
    X_norm = (X_true - lds.x_mean) / lds.x_std

    # Initialize storage for R² values
    r2_X = []
    r2_Z = []
    r2_Y = []

    print("Computing rollouts efficiently, one step at a time...")

    # Start with normalized states at time 0 (similar to plot_model_fit_pred.py line 65)
    Xhat_ahead = X_norm.copy()

    # For each prediction horizon, roll forward one more step
    for horizon in range(1, max_horizon + 1):
        # Apply one step of LDS dynamics: X[t+1] = A @ X[t] + B @ U[t]
        # (similar to plot_model_fit_pred.py line 74)
        Xhat_ahead = (
            Xhat_ahead @ lds.A.T + U_norm[horizon - 1 : horizon - 1 + T] @ lds.B.T
        )

        # Compute how many predictions we can make at this horizon
        n_predictions = T - horizon
        if n_predictions < 2:  # Need at least 2 samples for R²
            r2_X.append(np.nan)
            r2_Z.append(np.nan)
            r2_Y.append(np.nan)
            continue

        # Get predictions and ground truth for this horizon
        X_pred = Xhat_ahead[:n_predictions] * lds.x_std + lds.x_mean
        X_true_slice = X_true[horizon : horizon + n_predictions]

        # Map to behavioral outputs
        if hasattr(lds, "x2z"):
            Z_pred = lds.x2z(Xhat_ahead[:n_predictions]) * lds.z_std + lds.z_mean
        else:
            Z_pred = (Xhat_ahead[:n_predictions] @ lds.C.T) * lds.z_std + lds.z_mean
        Z_true_slice = Z_true[horizon : horizon + n_predictions]

        # Calculate R² scores
        r2_X_val = r2_score(X_true_slice, X_pred, multioutput="variance_weighted")
        r2_Z_val = r2_score(Z_true_slice, Z_pred, multioutput="variance_weighted")

        r2_X.append(r2_X_val)
        r2_Z.append(r2_Z_val)

        # Map to recording outputs if available
        if Y_true is not None and Y_true.shape[1] > 0:
            Y_pred = (Xhat_ahead[:n_predictions] @ lds.C.T) * lds.y_std + lds.y_mean
            Y_true_slice = Y_true[horizon : horizon + n_predictions]
            r2_Y_val = r2_score(Y_true_slice, Y_pred, multioutput="variance_weighted")
            r2_Y.append(r2_Y_val)
        else:
            r2_Y.append(np.nan)

        if horizon % 50 == 0 or horizon == max_horizon:
            print(
                f"Horizon {horizon}: {n_predictions} predictions, R2_X={r2_X_val:.3f}, R2_Z={r2_Z_val:.3f}"
            )

    # Create results dataframe
    results = {
        "i_step_ahead": np.arange(1, max_horizon + 1),
        "R2_X": r2_X,
        "R2_Z": r2_Z,
    }

    if Y_true is not None:
        results["R2_Y"] = r2_Y

    df = pd.DataFrame(results)

    # Save to CSV
    print(f"Saving rollout R² analysis to {outfile}")
    df.to_csv(outfile, index=False)
    print(f"Analysis complete. Saved {len(df)} rows of R² data.")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    lds_model = LDS.load(snakemake.input.lds)
    sys_id_data = ExptData.load(snakemake.input.sys_id)

    analyze_model_fit_rollout(
        snakemake.wildcards, lds_model, sys_id_data, snakemake.output[0]
    )

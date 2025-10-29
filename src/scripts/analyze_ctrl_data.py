import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from snakemake.script import snakemake

from src.log_redirector import redirect_output_to_file
from src.scripts.ctrl_expts import CtrlData
from src.scripts.model_fit import ModelFitPredData
from src.scripts.sys_id_expts import ExptData


def nrmse(R, X, Xbaseline):
    # to avoid having to collect baseline trials, just assume baseline "reference" is the mean
    errors = X - R
    baseline_errors = Xbaseline - Xbaseline.mean()
    return np.linalg.norm(errors) / (
        np.linalg.norm(baseline_errors) + 1e-10 * np.linalg.norm(errors)
    )


def analyze_ctrl_data(
    cond_params,
    ctrl_data: CtrlData,
    fsnn_data_analysis_file,
    model_fit_analysis_file,
    sys_id_data: ExptData,
    model_fit_preds: ModelFitPredData,
    outfile,
):
    results = {**cond_params}

    # R2
    results["R2_track_Zref_Z"] = r2_score(ctrl_data.Zref, ctrl_data.Z)
    results["R2_track_Xref_X_fsnn"] = r2_score(ctrl_data.Xref_fsnn, ctrl_data.X)
    results["R2_track_Xref_Xhat_fsnn"] = r2_score(
        ctrl_data.Xref_fsnn, ctrl_data.Xhat_fsnn
    )
    results["R2_track_Xref_Xhat_lds"] = r2_score(ctrl_data.Xref_lds, ctrl_data.Xhat_lds)

    # Q
    results["Q_track_Zref_Z"] = 1 - nrmse(ctrl_data.Zref, ctrl_data.Z, sys_id_data.Z)
    results["Q_track_Xref_X_fsnn"] = 1 - nrmse(
        ctrl_data.Xref_fsnn, ctrl_data.X, sys_id_data.X
    )
    results["Q_track_Xref_Xhat_fsnn"] = 1 - nrmse(
        ctrl_data.Xref_fsnn, ctrl_data.Xhat_fsnn, model_fit_preds.Xhat_lds2fsnn
    )
    results["Q_track_Xref_Xhat_lds"] = 1 - nrmse(
        ctrl_data.Xref_lds, ctrl_data.Xhat_lds, model_fit_preds.Xhat_lds
    )

    results["ctrl_compute_latency_ms"] = ctrl_data.compute_latency_ms

    results |= pd.read_csv(model_fit_analysis_file).iloc[0].to_dict()
    print(results)

    df = pd.DataFrame([results])
    print(f"Saving analysis results to {outfile}")
    df.to_csv(outfile, index=False)


if __name__ == "__main__" and "snakemake" in locals():
    with redirect_output_to_file(snakemake.log[0]):
        ctrl_data = CtrlData.load(snakemake.input.ctrl_data)
        sys_id_data = ExptData.load(snakemake.input.sys_id_data)
        model_fit_preds = ModelFitPredData.load(snakemake.input.model_fit_preds)
        analyze_ctrl_data(
            snakemake.wildcards,
            ctrl_data,
            snakemake.input.fsnn_data_analysis,
            snakemake.input.model_fit_analysis,
            sys_id_data,
            model_fit_preds,
            snakemake.output[0],
        )

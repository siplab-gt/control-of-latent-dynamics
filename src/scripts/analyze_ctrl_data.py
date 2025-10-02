import pandas as pd
from sklearn.metrics import r2_score
from snakemake.script import snakemake

from src.scripts.ctrl_expts import CtrlData
from src.scripts.sys_id_expts import ExptData


def analyze_ctrl_data(ctrl_data: CtrlData, model_fit_analysis_file, outfile):
    results = {}
    results["R2_ctrl_Z_Zref"] = r2_score(ctrl_data.Zref, ctrl_data.Z)
    results["R2_ctrl_Xhat_Xref_lds"] = r2_score(ctrl_data.Xref_lds, ctrl_data.Xhat_lds)
    results["R2_ctrl_Xhat_Xref_fsnn"] = r2_score(
        ctrl_data.Xref_fsnn, ctrl_data.Xhat_fsnn
    )

    results["ctrl_compute_latency_ms"] = ctrl_data.compute_latency_ms

    results |= pd.read_csv(model_fit_analysis_file).iloc[0].to_dict()
    print(results)

    df = pd.DataFrame([results])
    print(f"Saving analysis results to {outfile}")
    df.to_csv(outfile, index=False)


if __name__ == "__main__" and "snakemake" in locals():
    ctrl_data = CtrlData.load(snakemake.input[0])
    analyze_ctrl_data(ctrl_data, snakemake.input[1], snakemake.output[0])

import pandas as pd
from snakemake.script import snakemake

from src.scripts.ctrl_expts import CtrlData


def analyze_ctrl_data(ctrl_data: CtrlData, outfile):
    # TODO: implement analysis logic
    # compute R2...
    results = {}
    results["R2_ctrl_Z_Zref"]
    results["R2_ctrl_Xhat_Xref_braid"]

    pass


if __name__ == "__main__" and "snakemake" in locals():
    expt_data = CtrlData.load(snakemake.input[0])
    analyze_ctrl_data(expt_data, snakemake.output[0])

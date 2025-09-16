from dataclasses import dataclass

import numpy as np
from BRAID import BRAIDModel
from BRAID.MainModel import MainModel, MainModelPrepareArgs
from jaxtyping import Float, UInt
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.helpers import seed_everything
from src.scripts.sys_id_expts import ExptData


@dataclass
class BraidModel(BaseDataClass):
    pass


def fit_model(cond_params, base_params, expt_data_file, braid_model_outfile):
    seed = seed_everything(int(cond_params["seed"]), "fit_model")
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])

    expt_data = ExptData.load(expt_data_file)

    idSys: BRAIDModel = BRAIDModel()
    methodCode = (
        "BRAID_Cz1HL64U"  # Only Cz will be nonlinear, 1 Hidden layer with 64 Units.
    )
    args = MainModelPrepareArgs(methodCode)
    args["epochs"] = int(base_params["epochs"])
    n_x = int(cond_params["n_pcs"])  # just assume we know this
    idSys.fit(
        expt_data.Y.T,
        expt_data.Z.T,
        U=expt_data.U.T,
        nx=n_x,
        n1=2,
        n3=None,  # by default
        n_pre=None,
        args_base=args,
    )

    braid_model = BraidModel()
    braid_model.save(braid_model_outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    fit_model(
        snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0]
    )

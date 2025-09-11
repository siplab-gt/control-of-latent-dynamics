from dataclasses import dataclass

import brian2.only as b2
import cleo
import numpy as np
from jaxtyping import Float, UInt
from matplotlib import pyplot as plt

from src.data_classes import BaseDataClass
from src.helpers import seed_everything


@dataclass
class BraidModel(BaseDataClass):
    pass


def fit_model(cond_params, base_params, expt_data_file, braid_model_outfile):
    seed = seed_everything(int(cond_params["seed"]), "fit_model")
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])

    braid_model = BraidModel()
    braid_model.save(braid_model_outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    fit_model(
        snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0]
    )

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import false_discovery_control

from src.data_classes import BaseDataClass
from src.helpers import preprocess_params
from src.scripts.dependence_analysis import SEED, CMIEstimate


@dataclass
class CMIEstimateSet(BaseDataClass):
    setting: str
    task: str
    analysis: str
    n_shuffles: int
    n_bootstraps: int
    random_seed: int
    ep_results: list[CMIEstimate] = field(default_factory=list)


def main(cond_params, base_params, infiles, outfile):
    print(f"Aggregating {len(infiles)} files into {outfile}...")
    ep_results = []

    for f in infiles:
        cmi_estimate = CMIEstimate.load(f)
        ep_results.append(cmi_estimate)

    result_set = CMIEstimateSet(
        setting=cond_params.setting,
        task=cond_params.task,
        analysis=cond_params.analysis,
        n_shuffles=base_params["n_shuffle_boot"],
        n_bootstraps=base_params["n_shuffle_boot"],
        random_seed=SEED,
        ep_results=ep_results,
    )
    result_set.save(outfile)

    pvals = [entry.p_value for entry in result_set.ep_results]
    if pvals:
        qvals = false_discovery_control(pvals, method="bh")
        for entry, qval in zip(result_set.ep_results, qvals, strict=True):
            entry.q_value = float(qval)
            print(f"{entry.ep}: q = {entry.q_value}")
    print(pd.DataFrame([entry.summary_dict for entry in result_set.ep_results]))

    print(f"Saved aggregated CMI results to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(
        snakemake.wildcards,
        preprocess_params(snakemake.params),
        snakemake.input,
        snakemake.output[0],
    )

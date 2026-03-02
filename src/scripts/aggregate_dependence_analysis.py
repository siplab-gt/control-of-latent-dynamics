from dataclasses import dataclass, field

from src.data_classes import BaseDataClass
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
    print(f"Saved aggregated CMI results to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.input, snakemake.output[0])

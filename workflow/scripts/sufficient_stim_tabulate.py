def tabulate_sufficient_stim_results(cfg, infiles, outfile):
    import pandas as pd
    import numpy as np

    # realism sweeps
    print("realism levels:")
    sweep_conds = []
    for realism_level in np.linspace(
        0, 1, num=cfg["num_realism_levels"], endpoint=False
    ):
        realism_cond = {}
        for key in cfg["realism_features"]:
            realism_cond[key] = cfg[key][int(realism_level * len(cfg[key])]
            )
        sweep_conds.append(realism_cond)
        print(realism_cond)

if __name__ == '__main__' and 'snakemake' in locals():
    from snakemake.script import snakemake

    tabulate_sufficient_stim_results(snakemake.config, snakemake.input, snakemake.output[0])
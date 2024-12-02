from itertools import product

import numpy as np
import pandas as pd
import yaml


def main(cfg):
    grid_search = {k: cfg[k] for k in ["seed", "task"]}
    keys = grid_search.keys()
    values = (grid_search[key] for key in keys)
    base_conditions = [dict(zip(keys, combination)) for combination in product(*values)]

    ref_cfg = cfg["reference_configuration"]
    ref_cfg["m"] = "ground-truth"
    for cond in base_conditions:
        cond.update(ref_cfg)

    vars_to_sweep = [
        "latent_dim",
        "total_scale",
        "network_size_um",
        "beam_width_um",
        "beam_spacing",
        "connection_distance_penalty",
        "connection_l2_penalty",
        "connection_layer_penalty",
        "dales_law",
    ]
    # one-at-a-time sweeps
    sweep_conds = [{var: value} for var in vars_to_sweep for value in cfg[var]]

    # realism sweeps
    print("realism levels:")
    for realism_level in np.linspace(
        0, 1, num=cfg["num_realism_levels"], endpoint=False
    ):
        realism_cond = {}
        for key in cfg["realism_features"]:
            realism_cond[key] = cfg[key][int(realism_level * len(cfg[key]))]
        sweep_conds.append(realism_cond)
        print(realism_cond)
    # TODO: not sure how to incorporate num_tasks here
    # can store files either as task=reaching_integration or as
    # reaching_true/integration_true

    # =========== Q1 ===========
    # would like to binary search for sufficient k, but for now just do the whole sweep
    q1_conditions = [
        dict(cond, k=k, setting="q1")
        for cond in base_conditions
        for k in cfg["q1"]["k"]
    ]

    q1_conditions = [
        cond | sweep_cond for cond in q1_conditions for sweep_cond in sweep_conds
    ]

    # =========== Q2 ===========
    # optimal stim/rec ratio
    stim_ratios = np.linspace(
        cfg["q2"]["stim_ratio_min"],
        cfg["q2"]["stim_ratio_max"],
        cfg["q2"]["num_stim_ratio_levels"],
    )
    k = (stim_ratios * cfg["q2"]["channel_budget"]).round().astype(int)
    m = cfg["q2"]["channel_budget"] - k
    q2_conditions = [
        dict(cond, setting="q2", k=k_, m=m_)
        for cond in base_conditions
        for k_, m_ in zip(k, m)
    ]
    q2_conditions = [
        cond | sweep_cond for cond in q2_conditions for sweep_cond in sweep_conds
    ]

    # =========== Q3 ===========
    #

    # combine into dataframe
    all_conditions = q1_conditions + q2_conditions
    df = pd.DataFrame(all_conditions)
    print(df)
    print(df.info())
    df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.config)

elif __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)

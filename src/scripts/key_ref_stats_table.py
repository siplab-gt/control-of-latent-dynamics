import pandas as pd

from src.helpers import preprocess_params, setup_debugger


def main(cfg, base_params, setting, key_metrics_csv, outfile):
    df = pd.read_csv(key_metrics_csv)

    # the ref condition is repeated for each ep, so just pick one EP
    # to get those ref condition results out
    first_ep = "n_trials_sys_id"
    # convert to string since ep_value col is object dtype
    ref_val = str(cfg["reference_params"][first_ep])
    ref_df = df[(df["ep"] == first_ep) & (df.ep_value == ref_val)]

    cols = df.columns.tolist()
    metric_cols = cols[cols.index("ctrl_latency") + 1 :]

    stats = []
    for km in metric_cols:
        col_data = ref_df[km].dropna()
        stats.append(
            {
                "key_metric": km,
                "mean": col_data.mean(),
                "median": col_data.median(),
                "SEM": col_data.sem(),
            }
        )

    stats_df = pd.DataFrame(stats)
    assert not stats_df.isnull().values.any(), "\n" + str(stats_df)
    stats_df.to_csv(outfile, index=False)
    print(f"Wrote ref stats to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config.get("debug", False))

    main(
        snakemake.config,
        preprocess_params(snakemake.params),
        snakemake.wildcards["setting"],
        snakemake.input[0],
        snakemake.output[0],
    )

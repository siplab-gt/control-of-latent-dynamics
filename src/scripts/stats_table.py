import pandas as pd

from src.helpers import preprocess_params
from src.scripts.aggregate_dependence_analysis import CMIEstimateSet
from src.scripts.main_figs import r


def stats_table(
    control_results: CMIEstimateSet,
    key_results: CMIEstimateSet,
    output_csv: str,
    output_tex: str,
    params: dict,
):
    control_df = pd.DataFrame(
        [entry.summary_dict for entry in control_results.ep_results]
    )
    key_df = pd.DataFrame([entry.summary_dict for entry in key_results.ep_results])

    cols_to_keep = ["ep", "cmi", "p_value", "q_value"]
    control_df = control_df[cols_to_keep].copy()
    key_df = key_df[cols_to_keep].copy()

    control_df["ep"] = control_df["ep"].apply(r)
    key_df["ep"] = key_df["ep"].apply(r)

    key_metric = r(params["key_metrics"][0])
    control_metric = r(params["ctrl_metric"])
    combined = pd.concat(
        [key_df.set_index("ep"), control_df.set_index("ep")],
        axis=1,
        keys=[f"{key_metric}", f"{control_metric}"],
    ).reset_index()

    combined.to_csv(output_csv, index=False)

    def format_val(x, is_q_val=False):
        if not isinstance(x, (float, int)):
            return x
        if pd.isna(x):
            return x
        if abs(x) < 0.001 and x != 0:
            s = f"{x:.2e}"
        else:
            s = f"{x:.3f}"

        if is_q_val and x < 0.05:
            s = f"\\textbf{{{s}}}"
        return s

    for col in combined.columns:
        if col[0] == "ep" or col == "ep":
            continue
        is_q = col[1] == "q_value" if isinstance(col, tuple) else False
        combined[col] = combined[col].apply(lambda x: format_val(x, is_q))

    to_replace = {
        "ep": "Exploratory parameter",
        "cmi": "CMI (bits)",
        "p_value": "$p$-value",
        "q_value": "$q$-value",
    }
    combined.ep = combined.ep.str.replace("(μm)", "")
    combined.ep = combined.ep.str.replace("\n", " ")
    combined = combined.rename(
        columns={col: to_replace[col] for col in to_replace.keys()}
    )
    setting_alias = {
        "sufficient_stim": "stimulation dimensionality experiments",
        "optimal_stim_rec": "stimulation--recording balance experiments",
    }[params["setting"]]

    q = "q1" if "sufficient_stim" in params["setting"] else "q2"

    combined.to_latex(
        output_tex,
        index=False,
        escape=False,
        label=f"t:{q}-stats",
        caption=(
            f"Statistical analysis results for {setting_alias}, relating "
            f" {key_metric} and {control_metric} to exploratory parameters."
        ),
        column_format="l|ccc|ccc",
    )


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    stats_table(
        CMIEstimateSet.load(snakemake.input[0]),
        CMIEstimateSet.load(snakemake.input[1]),
        snakemake.output[0],
        snakemake.output[1],
        preprocess_params(snakemake.params) | dict(snakemake.wildcards),
    )

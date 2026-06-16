import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.rcsetup import cycler

c = {
    "dark": "#004d3d",
    "main": "#009988",
    "light": "#00ccaa",
    "a1": "#545bd7",
    "a2": "#eec03a",
    "a3": "#e34d30",
    "a4": "#be50a1",
    "opto1": "#72b5f2",
    "opto2": "#FF6145",
    "task": "#0F5FA8",
}
c["ctrl"] = c["main"]
c["key"] = c["a1"]
c["x"] = c["a2"]
c["y"] = c["a3"]
c["z"] = c["a4"]

ctrl_palette = {
    "OL": "0.4",
    "MPC": c["ctrl"],
    "MPC with latency": c["light"],
}
key_palette = ctrl_palette | {"MPC": c["key"]}

ctrl_styles = {
    "OL": "",
    "MPC": "",
    "MPC with latency": (2, 1),
}

cmap = sns.dark_palette(c["light"], reverse=True, as_cmap=True)
div_cmap = sns.color_palette("icefire_r", as_cmap=True)

double_col_width = 17 / 2.54

sns_errorbar_kwargs = dict(errorbar=("ci", 95), seed=19120731, n_boot=1000)


def style_plots(context="paper", fontscale=5 / 6):
    """fontscale=5/6 goes from default paper font size of 9.6 down to 8"""

    plt.style.use("default")

    try:
        import seaborn as sns

        sns.set_context(context, font_scale=fontscale)
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")
        warnings.warn("Seaborn not found, using matplotlib style, ignoring fontscale")
    # keep white background for now, makes viewing and sharing easier, though harder for figure assembly
    plt.rc("savefig", transparent=False, bbox="tight", dpi=300)
    plt.rc("svg", fonttype="none")
    plt.rc("axes.spines", top=False, right=False)
    plt.rc(
        "font",
        **{"sans-serif": "Source Sans Pro"},
    )
    plt.rc(
        "axes",
        prop_cycle=cycler("color", [c["main"], c["a1"], c["a2"], c["a3"], c["a4"]]),
    )


if __name__ == "__main__":
    style_plots()

    # create CSV with all relevant values for use elsewhere
    import pandas as pd
    from matplotlib import font_manager as fm

    rows = []
    for cname, cval in c.items():
        rows.append({"name": f"c_{cname}", "value": cval})

    # from rcParams: font sizes, line widths
    for param_name in [
        "font.size",
        "figure.titlesize",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "lines.linewidth",
        "axes.linewidth",
    ]:
        value = plt.rcParams[param_name]
        if value in fm.font_scalings.keys():
            value = fm.font_scalings[value] * plt.rcParams["font.size"]
        rows.append({"name": param_name, "value": value})

    pd.DataFrame(rows).to_csv("results/plot-style-params.csv", index=False)
    print("Created results/plot-style-params.csv")

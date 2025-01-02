import warnings

import matplotlib.pyplot as plt
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
}


def style_plots(fontscale=5 / 6):
    """fontscale=5/6 goes from default paper font size of 9.6 down to 8"""

    plt.style.use("default")

    try:
        import seaborn as sns

        sns.set_context("paper", font_scale=fontscale)
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

"""Plot data from fSNN simulations. Mainly for debugging."""

import brian2.only as b2
import matplotlib.pyplot as plt
import numpy as np

import src.plot_style as sty
from src.helpers import load_pkl_gz


def main(data_file, out_file):
    sty.style_plots(fontscale=1.5)
    plt.switch_backend("agg")
    data = load_pkl_gz(data_file)

    if "kdm1_mon" in data:
        kdm1 = data["kdm1_mon"]["kdm1"]
        assert np.allclose(np.diff(kdm1, axis=1), 0), (
            "kdm1 should be constant across neurons"
        )
    assert not np.allclose(data["w_mon"]["wLeft"], data["w_mon"]["wRight"]), (
        "s_mon wLeft and wRight should differ"
    )

    # add mse_y and mse_z to data
    data["mse_y"] = {
        "t": data["y_mon"]["t"],
        "mse_y": np.linalg.norm(data["y_mon"]["y_"] - data["y_mon"]["ytilde"], axis=1)
        / data["y_mon"]["y_"].shape[1],
    }
    data["mse_z"] = {
        "t": data["z_mon"]["t"],
        "mse_z": np.linalg.norm(data["z_mon"]["z"] - data["z_mon"]["fout"], axis=1)
        / data["z_mon"]["z"].shape[1],
    }

    n_series = 0
    for vars in data.values():
        for var in vars:
            if var not in ["t", "N", "count"]:
                n_series += 1

    T_s = data[next(iter(data))]["t"][-1] / b2.second

    fig, axs = plt.subplots(
        n_series, 1, figsize=(2 * T_s, 2 * n_series), layout="compressed", sharex=True
    )

    i_series = 0
    for mon, states in data.items():
        t_s = states["t"] / b2.second
        for var_name, values in states.items():
            if var_name in ["t", "N", "count"]:
                continue
            assert np.all(~np.isnan(values)), f"Found NaNs in {mon}.{var_name}"
            if isinstance(values, b2.Quantity):
                u = values.get_best_unit()
            else:
                u = 1.0
            ax = axs[i_series] if n_series > 1 else axs
            print(f"Plotting {mon}.{var_name} with unit {u}")
            print(f"  {t_s.shape=}, {values.shape=}")
            u_label = str(u) if u != 1.0 else "a.u."
            if mon == "spk_mon":
                ax.plot(t_s, values / u, ".", ms=0.2)
                u_label = "neuron index"
            else:
                ax.plot(t_s, values / u)
            ax.set(title=f"{mon}.{var_name}", ylabel=f"{mon}.{var_name} ({u_label})")
            i_series += 1
    ax.set_xlabel("t (s)")

    print(f"Saving figure to {out_file}")
    fig.savefig(out_file)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.input[0], snakemake.output[0])

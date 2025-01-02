import brian2.only as b2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection, PatchCollection

from src.interface_array import get_interface_array
from src.plot_style import c, style_plots


def main(cond_params, base_params, outfiles):
    style_plots()
    fig, ax = plt.subplots()

    side_length_mm = np.sqrt(float(base_params["array_side_length_mm"])) ** 2 * float(
        cond_params["network_area_scale"]
    )
    net_rect = plt.Rectangle(
        (-side_length_mm / 2, -side_length_mm / 2),
        side_length_mm,
        side_length_mm,
        # fill=False,
        facecolor=c["main"] + "16",
        edgecolor=c["main"] + "FF",
        label="network",
    )
    ax.add_patch(net_rect)

    grid = get_interface_array(
        float(base_params["array_side_length_mm"]),
        float(cond_params["total_scale"]),
        int(base_params["array_n_per_side"]),
    )
    if cond_params["setting"] == "sufficient_stim":
        from src.cleo_prep_sufficient_stim import light_and_probe_coords
    elif cond_params["setting"] == "optimal-stim-rec":
        from src.cleo_prep_optimal_stim_rec import light_and_probe_coords

    light_coords, light_groups, probe_coords = light_and_probe_coords(
        grid, int(cond_params["k"])
    )
    # TODO: two opsins/lights
    beam_width_mm = float(cond_params["beam_width_um"]) / 1000
    ellipses = EllipseCollection(
        beam_width_mm,
        beam_width_mm,
        0,
        units="xy",
        offsets=light_coords[:, :2] / b2.mm,
        transOffset=ax.transData,
        facecolors=c["opto1"],
        edgecolors="none",
        # alpha=0.5,
        label="light array",
    )
    ax.add_collection(ellipses)
    for i_group, group in enumerate(light_groups):
        for i_channel in group:
            ax.text(
                *(light_coords[i_channel, :2] / b2.mm),
                f"{i_group + 1}",
                fontsize=6,
                ha="center",
            )

    if probe_coords:
        ax.plot(
            probe_coords[:, 0] / b2.mm, probe_coords[:, 1] / b2.mm, "x", label="probe"
        )

    xy_lim = (side_length_mm / 2) * 1.2
    ax.set(
        xlim=(-xy_lim, xy_lim),
        ylim=(-xy_lim, xy_lim),
        aspect="equal",
        xlabel="x (mm)",
        ylabel="y (mm)",
    )

    # NOTE: light not in legend
    ax.legend()
    for f in outfiles.top:
        fig.savefig(f)

    # side view
    fig, ax = plt.subplots()

    layer_bounds_mm = [float(lb) / 1000 for lb in base_params["layer_boundaries_um"]]
    density_ratios = np.array([float(dr) for dr in base_params["layer_density_ratios"]])
    base_neuron_density = float(base_params["base_neuron_density"])
    adj_density_ratios = 1 + (density_ratios - 1) * float(cond_params["layeredness"])
    layer_densities = base_neuron_density * adj_density_ratios

    layer_rects = []
    net_alpha_max = 0.3
    layer_alphas = net_alpha_max * layer_densities / layer_densities.max()
    for i, layer_name in enumerate(["L2/3", "L4", "L5"]):
        layer_rects.append(
            plt.Rectangle(
                (-side_length_mm / 2, layer_bounds_mm[i]),
                side_length_mm,
                layer_bounds_mm[i + 1] - layer_bounds_mm[i],
                facecolor=c["main"] + hex(int(layer_alphas[i] * 255))[2:],
                edgecolor=c["main"] + "FF",
                linewidth=0.5,
                label=layer_name,
            )
        )
        ax.add_patch(layer_rects[-1])
    legend_handles = layer_rects

    light_rects = []
    for light_coord in light_coords:
        light_rects.append(
            plt.Rectangle(
                light_coord[[0, 2]] / b2.mm,
                beam_width_mm,
                beam_width_mm * 2,
                facecolor=c["opto1"],
                edgecolor="none",
                label="light array",
            )
        )
    ax.add_collection(
        PatchCollection(
            light_rects,
            facecolor=c["opto1"],
            edgecolor="none",
        )
    )
    legend_handles.append(light_rects[0])

    if probe_coords:
        probe_markers = ax.plot(
            probe_coords[:, 0] / b2.mm, probe_coords[:, 2] / b2.mm, "x", label="probe"
        )
        legend_handles.append(probe_markers)

    ax.set(
        xlim=(-xy_lim, xy_lim),
        ylim=(layer_bounds_mm[-1] * 1.1, 0),
        xlabel="x (mm)",
        ylabel="depth (mm)",
        aspect="equal",
    )
    ax.legend(handles=legend_handles)

    for f in outfiles.side:
        fig.savefig(f)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.output)

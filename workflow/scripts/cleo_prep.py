import brian2.only as b2
import cleo
import numpy as np

from src.data_classes import CleoExperiment, fSNN
from src.grouped_light import GroupedLight


def main(cond_vars, params, fsnn_file, outfile):
    fsnn = fSNN.load(fsnn_file)
    light_model = cleo.light.fiber473nm(
        R0=float(cond_vars["beam_width_um"]) * b2.um / 2
    )
    grid_len = float(params["array_side_length_mm"]) * float(cond_vars["total_scale"])
    # TODO: define grid size by beam spacing?
    x, y, z = np.meshgrid(
        np.linspace(-grid_len / 2, grid_len / 2, int(params["array_n_per_side"])),
        np.linspace(-grid_len / 2, grid_len / 2, int(params["array_n_per_side"])),
        0.6,  # TODO: what depth? just one layer? this is L4
        indexing="ij",
    )
    grid = np.stack((x, y, z), axis=-1).reshape(-1, 3) * b2.mm

    # separate files so one can change without causing the other to be rerun
    if cond_vars["setting"] == "sufficient_stim":
        from src.cleo_prep_sufficient_stim import light_and_probe_coords
    elif cond_vars["setting"] == "optimal-stim-rec":
        from src.cleo_prep_optimal_stim_rec import light_and_probe_coords

    light_coords, light_groups, probe_coords = light_and_probe_coords(
        grid, int(cond_vars["k"])
    )

    array = GroupedLight(
        name="light array",
        coords=light_coords,
        groups=light_groups,
        light_model=light_model,
    )
    sim = cleo.CLSimulator(fsnn.net)
    sim.inject(array, fsnn.ng)

    if probe_coords:
        probe = cleo.ephys.Probe(probe_coords)
        sim.inject(probe, fsnn.ng)
    else:
        probe = None

    cleo_expt = CleoExperiment(fsnn, array, probe, sim)
    cleo_expt.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0])

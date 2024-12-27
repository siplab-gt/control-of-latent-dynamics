import brian2.only as b2
import cleo
import numpy as np

from src.data_classes import CleoExperiment, fSNN
from src.grouped_light import GroupedLight


def main(cond_vars, cfg, fsnn_file, outfile):
    fsnn = fSNN.load(fsnn_file)
    light_model = cleo.light.fiber473nm(
        R0=float(cond_vars["beam_width_um"]) * b2.um / 2
    )
    grid_len = float(cfg["array_side_length_mm"]) * float(cond_vars["total_scale"])
    # TODO: define grid size by beam spacing?
    x, y, z = np.meshgrid(
        np.linspace(-grid_len / 2, grid_len / 2, int(cfg["array_n_per_side"])),
        np.linspace(-grid_len / 2, grid_len / 2, int(cfg["array_n_per_side"])),
        0.6,  # TODO: what depth? just one layer? this is L4
        indexing="ij",
    )
    grid = np.stack((x, y, z), axis=-1).reshape(-1, 3) * b2.mm
    if cond_vars["setting"] == "sufficient-stim":
        coords = grid
        groups = np.arange(len(coords)).reshape(int(cond_vars["k"]), -1)
        probe = None

    array = GroupedLight(
        name="light array", coords=coords, groups=groups, light_model=light_model
    )
    sim = cleo.CLSimulator(fsnn.net)
    sim.inject(array, fsnn.ng)
    if probe:
        sim.inject(probe, fsnn.ng)
    cleo_expt = CleoExperiment(fsnn, array, probe, sim)
    cleo_expt.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.config, snakemake.input[0], snakemake.output[0])

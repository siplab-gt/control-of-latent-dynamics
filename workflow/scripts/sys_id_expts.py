import brian2.only as b2
import cleo
import numpy as np

from src.data_classes import CleoPrep, ExptData, fSNN
from src.grouped_light import GroupedLight
from src.helpers import seed_everything


def run_sys_id_expts(cond_params, base_params, cleo_prep_file, outfile):
    print(f"{base_params=}")
    print(base_params.keys())
    seed = seed_everything(int(cond_params["seed"]), "sys_id_expts")
    rng = np.random.default_rng(seed)

    k = int(cond_params["k"])
    u_max = float(base_params["u_max_mWmm2"])
    try:
        print(f"{base_params.t_intertrial_ms=}")
    except:
        print("no base_params.t_intertrial_ms")
        # TODO: unpack params
        pass

    class SysIDProcessor(cleo.LatencyIOProcessor):
        def process(self, state_dict, t_samp):
            return {"light array": rng.uniform(0, u_max, k) * b2.mwatt / b2.mm2}, t_samp

    cleo_prep = CleoPrep.load(cleo_prep_file)
    cleo_prep.sim.set_io_processor(SysIDProcessor())
    print("will print state mon keys")
    print(f"{cleo_prep.fsnn.state_mon.Y=}")
    cleo_prep.sim.run(
        1 * b2.ms,  # TODO
        report_period=50 * b2.second,
    )
    print("successfully ran")

    expt_data = ExptData(
        U=cleo_prep.light.irradiance_,
        Y=cleo_prep.fsnn.state_mon.Y,
    )
    expt_data.save(outfile)


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    run_sys_id_expts(
        snakemake.wildcards, snakemake.params, snakemake.input[0], snakemake.output[0]
    )

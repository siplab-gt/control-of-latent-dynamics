import brian2.only as b2

from src.fsnn import prep_fsnn
from src.helpers import seed_everything


def main(cond_params, base_params, outfile):
    fsnn = prep_fsnn(cond_params, base_params)
    seed_everything(int(cond_params["seed"]), "train_fsnn")
    b2.prefs.codegen.target = "numpy"
    fsnn.net.run(1 * b2.ms)
    print(fsnn)
    fsnn.net.store(filename=outfile)
    print(f"saved to {outfile}")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.wildcards, snakemake.params, snakemake.output[0])

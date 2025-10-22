import brian2.only as b2

from src.fsnn import run_main
from src.helpers import preprocess_params, setup_logging
from src.log_redirector import redirect_output_to_file

if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    with redirect_output_to_file(snakemake.log[0]):
        setup_logging(snakemake.config)
        fsnn = run_main(
            preprocess_params(snakemake.wildcards),
            preprocess_params(snakemake.params[0]),
            snakemake.config["train_device"],
            None,  # no infile, start fresh
            snakemake.output[0],
            mode="demean",
            record=False,
        )
        b2.device.delete()

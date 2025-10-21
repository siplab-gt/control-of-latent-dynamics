from src.fsnn import run_main
from src.helpers import preprocess_params
from src.log_redirector import redirect_output_to_file

if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    with redirect_output_to_file(snakemake.log[0]):
        fsnn = run_main(
            preprocess_params(snakemake.wildcards),
            preprocess_params(snakemake.params[0]),
            snakemake.config["train_device"],
            snakemake.input[0],  # demeaned
            snakemake.output[0],  # trained
            mode="train",
            record=False,
        )

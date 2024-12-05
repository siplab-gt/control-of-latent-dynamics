def main(cfg, infiles, outfile):
    with open(outfile, "w") as f:
        f.write("File created successfully.\n")


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    main(snakemake.config, snakemake.input, snakemake.output[0])

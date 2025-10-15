from dataclasses import dataclass, field

import brian2.only as b2

from src.fsnn import run_main
from src.helpers import numerize_and_attrify
from src.log_redirector import redirect_output_to_file


@dataclass
class fSNNTestData:
    """Data from testing the fSNN."""

    t_ms: b2.Quantity = field(default_factory=list)
    """Times of state monitor recordings in ms, list of arrays"""
    X: list = field(default_factory=list)
    """Recorded factors, shape (T, n_factors)"""
    v: b2.Quantity = field(default_factory=list)
    """Recorded membrane potentials, shape (T, n_recorded_neurons)"""


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    with redirect_output_to_file(snakemake.log[0]):
        fsnn = run_main(
            numerize_and_attrify(snakemake.wildcards),
            numerize_and_attrify(snakemake.params[0]),
            snakemake.config["train_device"],
            snakemake.input[0],
            None,
            mode="train",
            record=True,
        )

        outfile = snakemake.output[0]
        data = fSNNTestData(
            t_ms=fsnn.state_mon.t / b2.ms,
            X=fsnn.state_mon.X,
            v=fsnn.v_mon.v / b2.mV,
        )
        data.save(outfile)
        print(f"saved fSNN test data to {outfile}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helpers import preprocess_params, setup_debugger
from src.plot_style import c, style_plots


def plot_r2_vs_horizon(
    r2_data_file,
    outfiles,
):
    """
    Create line plot showing R² degradation vs prediction horizon

    Args:
        r2_data_file: CSV file with R² vs horizon data
        outfiles: Output file paths
        cond_params: Condition parameters for title/annotations
    """
    style_plots()

    # Load R² data
    df = pd.read_csv(r2_data_file)

    if df.empty:
        print("Warning: Empty R² data file")
        return

    # Determine which output types are available
    available_outputs = []
    labels = []
    colors = []

    if "R2_X" in df.columns and not df["R2_X"].isna().all():
        available_outputs.append("R2_X")
        labels.append("Latent states (X)")
        colors.append(c["x"])

    if "R2_Z" in df.columns and not df["R2_Z"].isna().all():
        available_outputs.append("R2_Z")
        labels.append("Behavioral outputs (Z)")
        colors.append(c["y"])

    if "R2_Y" in df.columns and not df["R2_Y"].isna().all():
        available_outputs.append("R2_Y")
        labels.append("Recordings (Y)")
        colors.append(c["z"])

    if not available_outputs:
        print("Warning: No valid R² data found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot each output type
    for output_col, label, color in zip(available_outputs, labels, colors):
        # Remove NaN values for plotting
        valid_mask = ~df[output_col].isna()
        if valid_mask.any():
            ax.plot(
                df.loc[valid_mask, "i_step_ahead"],
                df.loc[valid_mask, output_col],
                label=label,
                color=color,
            )

    # Formatting
    ax.set(
        xlabel="Prediction horizon (steps ahead)",
        ylabel="R² score",
        title="Model Fit R² vs Prediction Horizon",
        ylim=(0, 1.05),
    )

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc="upper right")

    # Save to all output files
    for outfile in outfiles:
        print(f"Saving R² plot to {outfile}")
        fig.savefig(outfile)

    plt.close()


if __name__ == "__main__" and "snakemake" in locals():
    from snakemake.script import snakemake

    setup_debugger(snakemake.config["debug"])
    style = snakemake.wildcards.get("style", "default")

    plot_r2_vs_horizon(
        snakemake.input.r2_data,
        snakemake.output,
    )

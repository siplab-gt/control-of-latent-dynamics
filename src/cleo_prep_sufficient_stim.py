"""Defines how light and probe coordinates are distributed.

Broken into a separate file so Snakemake can only rerun necessary changed code."""

import logging

import brian2.only as b2
import numpy as np
from brian2 import Quantity, mm
from jaxtyping import Float
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


def light_and_probe_coords(
    grid: Quantity, k: int, m: int, rng: np.random.Generator, **kwargs
) -> tuple[Quantity, list[list[int]], Quantity]:
    # only going to be light coords, since no probe on sufficient_stim

    # Use k-means clustering to partition grid into k spatially contiguous groups
    grid_mm = grid / mm  # Convert to dimensionless coordinates for clustering

    # Extract integer seed from rng for KMeans
    random_state = rng.integers(0, 2**31 - 1) if rng is not None else 42
    # kmeans = KMeans(n_clusters=k, init="random", random_state=random_state, n_init=10)

    # Fit k-means and get cluster assignments for each grid point
    # labels = kmeans.fit_predict(grid_mm)
    labels = ccvt_grid(grid_mm, k, rng, max_iter=20)

    # Organize indices into groups based on cluster assignments
    groups = [np.where(labels == i)[0].tolist() for i in range(k)]

    # no probe on sufficient_stim
    assert m == "NA" or m == 0

    return grid, groups, None


def ccvt_grid(grid_mm, k, rng, max_iter=20):
    """constrainted centroidal voronoi tessellation for grid partitioning"""
    # 1. Initialize Grid Coordinates (N x 2)
    points = grid_mm
    N = points.shape[0]

    # 2. Initialize Centroids (Poisson Disk or Random)
    # For simplicity: random choice from grid points
    indices = rng.choice(N, k, replace=False)
    centroids = points[indices].astype(float)

    # Define capacities (balanced as possible)
    counts = np.full(k, N // k)
    counts[: N % k] += 1

    # 3. Iterative Optimization
    for i in range(max_iter):
        # Calculate N x k distance matrix
        dist_mat = cdist(points, centroids, metric="sqeuclidean")

        # Expand distance matrix to account for capacities
        # We repeat each centroid column by its capacity
        expanded_dist_mat = np.repeat(dist_mat, counts, axis=1)

        # Solve Linear Sum Assignment (Optimal Transport)
        # This finds the best matching given the capacity constraints
        _, col_ind = linear_sum_assignment(expanded_dist_mat)

        # Map back to original centroid indices
        # If capacities are all 2, col_ind [0,1,2,3] -> [0,0,1,1]
        cluster_labels = np.repeat(np.arange(k), counts)[col_ind]

        # 4. Update Centroids
        new_centroids = np.array(
            [points[cluster_labels == j].mean(axis=0) for j in range(k)]
        )

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    print(f"CCVT converged in {i + 1} iterations")
    return cluster_labels


def visualize_grid_partitions(labels: np.ndarray, k: int, grid_size: int = 10) -> None:
    """Visualize grid partitions as a colored table.

    Args:
        labels: Array of cluster assignments for each grid point
        k: Number of clusters
        grid_size: Side length of grid (default 10 for 10x10)

    Returns:
        Table: Rich Table object (not printed, for layout composition)
    """
    # Color palette for clusters
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "magenta",
        "cyan",
        "dark_red",
        "dark_blue",
        "dark_green",
        "dark_yellow",
        "dark_magenta",
        "dark_cyan",
    ]

    table = Table(title=f"Grid Partitions (k={k})", show_header=False, padding=(0, 1))

    # Add grid_size columns with no header
    for _ in range(grid_size):
        table.add_column()

    # Build each row
    for row_idx in range(grid_size):
        row_cells = []
        for col_idx in range(grid_size):
            grid_idx = row_idx * grid_size + col_idx
            cluster = labels[grid_idx]
            color = colors[cluster % len(colors)]
            # Show cluster index with matching background color
            cell = f"[{color} on {color}]{cluster:2d}[/{color} on {color}]"
            row_cells.append(cell)
        table.add_row(*row_cells)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    from src.interface_array import get_interface_array

    # simple test
    rng = np.random.default_rng(42)
    grid = get_interface_array(
        depth_mm=1.2, array_side_length_mm=4.0, array_n_per_side=10
    )
    for k in [1, 19, 50, 83, 100]:
        light_coords, light_groups, probe_coords = light_and_probe_coords(
            grid, k, m=0, rng=rng
        )

        # Get labels for visualization
        labels = np.zeros(len(grid), dtype=int)
        for group_idx, group_indices in enumerate(light_groups):
            labels[group_indices] = group_idx

        visualize_grid_partitions(labels, k)

        # Print group size distribution summary
        group_sizes = [len(group) for group in light_groups]
        size_counts = {}
        for size in group_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1

        print(f"Group size distribution: ", end="")
        print(
            ", ".join(f"{count}×{size}" for size, count in sorted(size_counts.items()))
        )

# tetris_packer/cli.py
"""
Command-line entry point for the Tetris Packing challenge.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

from . import shapes
from . import solver_greedy


def save_grid_csv(grid: np.ndarray, path: str) -> None:
    """Save a grid (2D numpy array) as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in grid.tolist():
            writer.writerow(row)


def visualize_grid(grid: np.ndarray, title: str = "Packed Grid") -> None:
    """Show the packed grid with different colors for each block id."""
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap("tab20")  # 20 distinct colors
    unique_ids = np.unique(grid)
    for pid in unique_ids:
        if pid == 0:
            continue
        mask = (grid == pid)
        # color index cycles through cmap
        color = cmap(pid % 20)
        y, x = np.where(mask)
        plt.scatter(x, -y, c=[color], s=200, marker="s")  # flip y for proper orientation
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optimal Block Packing (Tetris Challenge)")
    parser.add_argument("input", help="Path to input JSON file (Option 1 format).")
    parser.add_argument("--output", "-o", default="packed_grid.csv", help="Output CSV file path.")
    parser.add_argument("--time", type=float, default=5.0, help="Time limit per width (seconds).")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization.")
    args = parser.parse_args(argv)

    # --- Load shapes ---
    try:
        input_data = shapes.load_shapes_from_json(args.input)
    except Exception as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        sys.exit(1)

    pieces = shapes.expand_shapes_from_input(input_data, allow_reflect=False)
    print(f"Loaded {len(pieces)} pieces from {args.input}")

    # --- Solve ---
    try:
        H, W, aspect, grid = solver_greedy.search_best_packing(pieces, time_per_width=args.time)
    except Exception as e:
        print(f"Solver failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Output results ---
    print(f"Rectangle: {H} x {W}")
    print(f"Aspect ratio: {aspect:.2f}")
    save_grid_csv(grid, args.output)
    print(f"Saved packed grid to {args.output}")

    # --- Visualization ---
    if not args.no_viz:
        visualize_grid(grid, f"Packed {H}x{W}, Aspect={aspect:.2f}")


if __name__ == "__main__":
    main()

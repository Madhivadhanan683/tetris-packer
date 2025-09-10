# tetris_packer/cli.py
"""
Command-line entry point for the Tetris Packing challenge.
Supports JSON input or interactive block input.
Optional animation of block placement.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

from . import shapes
from . import solver_greedy
from . import placer
from .visualize import animate_packing

# --- CSV / Grid utilities ---
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
        color = cmap(pid % 20)
        y, x = np.where(mask)
        plt.scatter(x, -y, c=[color], s=200, marker="s")  # flip y-axis
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.show()

# --- Interactive input ---
def input_blocks_interactively():
    B = int(input("Enter number of block types (B): "))
    blocks = []
    for i in range(B):
        print(f"\n--- Block {i+1} ---")
        copies = int(input("Number of copies: "))
        rows = int(input("Number of rows in the block: "))
        cols = int(input("Number of columns in the block: "))
        print("Enter the block row by row, using 1 for filled and 0 for empty:")
        matrix = []
        for r in range(rows):
            row_input = input(f"Row {r+1}: ")
            row = [int(x) for x in row_input.strip().split()]
            if len(row) != cols:
                raise ValueError(f"Expected {cols} columns, got {len(row)}")
            matrix.append(row)
        name = input("Block name (optional): ").strip() or f"shape_{i}"
        blocks.append({"copies": copies, "matrix": matrix, "name": name})
    input_data = {"B": B, "blocks": blocks}
    pieces = shapes.expand_shapes_from_input(input_data)
    return pieces

# --- Extract placements for animation ---
def extract_placements(grid: np.ndarray, pieces) -> list:
    """
    Reconstruct approximate placements of each piece for animation.
    Returns list of (block_matrix, row, col, block_name)
    """
    placements = []
    for piece in pieces:
        # locate top-left of piece.id in grid
        positions = np.argwhere(grid == piece.id)
        if positions.size == 0:
            continue
        y_min, x_min = positions.min(axis=0)
        placements.append((piece.base, y_min, x_min, piece.name))
    return placements

# --- Main CLI ---
def main(argv=None):
    parser = argparse.ArgumentParser(description="Optimal Block Packing (Tetris Challenge)")
    parser.add_argument("input", nargs="?", default=None, help="Path to input JSON file")
    parser.add_argument("--output", "-o", default="packed_grid.csv", help="Output CSV file path")
    parser.add_argument("--time", type=float, default=5.0, help="Time limit per width (seconds)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--animate", action="store_true", help="Animate block placement")
    parser.add_argument("--interactive", action="store_true", help="Input blocks interactively")
    args = parser.parse_args(argv)

    # --- Load shapes ---
    if args.interactive:
        try:
            pieces = input_blocks_interactively()
        except Exception as e:
            print(f"Interactive input error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.input:
            print("JSON input file required if not using --interactive", file=sys.stderr)
            sys.exit(1)
        try:
            input_data = shapes.load_shapes_from_json(args.input)
            pieces = shapes.expand_shapes_from_input(input_data, allow_reflect=False)
        except Exception as e:
            print(f"Error loading input file: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(pieces)} pieces")

    # --- Solve packing ---
    try:
        H, W, aspect, grid = solver_greedy.search_best_packing(pieces, time_per_width=args.time)
    except Exception as e:
        print(f"Solver failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Output ---
    print(f"Rectangle: {H} x {W}")
    print(f"Aspect ratio: {aspect:.2f}")
    save_grid_csv(grid, args.output)
    print(f"Saved packed grid to {args.output}")

    # --- Visualization ---
    if not args.no_viz:
        visualize_grid(grid, f"Packed {H}x{W}, Aspect={aspect:.2f}")

    # --- Animation ---
    if args.animate:
        try:
            placements = extract_placements(grid, pieces)
            animate_packing(placements, H, W)
        except Exception as e:
            print(f"Animation failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

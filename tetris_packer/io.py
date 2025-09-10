import json
import csv
from pathlib import Path
import numpy as np

def load_json(path):
    """Load block data from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "blocks" not in data:
        raise ValueError("Invalid JSON: missing 'blocks' key")

    return data


def save_csv(grid, path):
    """Save a 2D numpy array as CSV."""
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in grid:
            writer.writerow(row)


def grid_from_csv(path):
    """Load grid back from CSV into numpy array."""
    path = Path(path)
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = [[int(x) for x in row] for row in reader]
    return np.array(rows)

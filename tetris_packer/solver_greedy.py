# tetris_packer/solver_greedy.py
"""
Greedy + backtracking solver for the Tetris packing problem.

Tries to arrange all given pieces inside a rectangular grid
so that the aspect ratio (max(H/W, W/H)) is minimized.
"""

from typing import List, Tuple, Optional
import numpy as np
import math
import time

from .shapes import Piece
from . import placer


__all__ = [
    "pack_with_width",
    "search_best_packing",
]


def pack_with_width(
    pieces: List[Piece],
    width: int,
    time_limit: float = 5.0,
) -> Optional[np.ndarray]:
    """
    Try to pack all pieces into a grid with fixed width.
    Height is computed as ceil(total_area / width) + margin.
    Uses backtracking search.

    Returns:
        - grid (np.ndarray) with piece ids placed
        - None if no valid packing was found within time limit
    """
    total_area = sum(p.area for p in pieces)
    est_height = math.ceil(total_area / width)
    # add a margin in case greedy doesn't fit perfectly
    height = est_height + 4

    grid = np.zeros((height, width), dtype=np.int32)
    # Sort pieces by area (biggest first)
    sorted_pieces = sorted(pieces, key=lambda p: -p.area)

    deadline = time.time() + time_limit

    def backtrack(i: int, g: np.ndarray) -> Optional[np.ndarray]:
        if time.time() > deadline:
            return None
        if i == len(sorted_pieces):
            return g
        piece = sorted_pieces[i]
        for variant in piece.variants:
            ph, pw = variant.shape
            for y in range(height - ph + 1):
                for x in range(width - pw + 1):
                    if placer.fits_at(g, variant, x, y):
                        g2 = placer.place_on(g, variant, x, y, piece.id, inplace=False)
                        result = backtrack(i + 1, g2)
                        if result is not None:
                            return result
        return None

    return backtrack(0, grid)


def search_best_packing(
    pieces: List[Piece],
    time_per_width: float = 5.0,
) -> Tuple[int, int, float, np.ndarray]:
    """
    Try different candidate widths and select the best aspect ratio packing.

    Returns: (H, W, aspect_ratio, grid)
    Raises: RuntimeError if no packing was found.
    """
    total_area = sum(p.area for p in pieces)
    if total_area <= 0:
        raise ValueError("Total area must be > 0")

    best = None
    sqrt_area = int(math.sqrt(total_area))

    # Candidate widths around sqrt(area)
    candidates = list(range(max(1, sqrt_area - 3), sqrt_area + 4))

    for w in candidates:
        grid = pack_with_width(pieces, w, time_limit=time_per_width)
        if grid is None:
            continue
        h, w2 = placer.bounding_box(grid)
        if h == 0 or w2 == 0:
            continue
        aspect = max(h / w2, w2 / h)
        if best is None or aspect < best[2]:
            best = (h, w2, aspect, placer.crop_to_bounding_box(grid))

    if best is None:
        raise RuntimeError("No valid packing found within time limits.")

    return best

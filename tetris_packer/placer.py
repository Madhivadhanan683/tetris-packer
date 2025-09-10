# tetris_packer/placer.py
from typing import Tuple, Optional
import numpy as np

__all__ = [
    "fits_at",
    "place_on",
    "remove_from",
    "bounding_box",
    "crop_to_bounding_box",
]

def fits_at(grid: np.ndarray, piece: np.ndarray, x: int, y: int) -> bool:
    """
    Check if `piece` (binary numpy array) can be placed at top-left coordinate (x,y)
    on `grid` without overlap and fully inside the bounds.
    - grid: 2D int numpy array (0 -> empty, >0 -> occupied by id)
    - piece: 2D binary uint8 array where 1 indicates filled cell
    - x,y: column (x) and row (y) of top-left placement
    Returns True if the piece fits.
    """
    if piece.ndim != 2 or grid.ndim != 2:
        raise ValueError("grid and piece must be 2D numpy arrays.")
    ph, pw = piece.shape
    H, W = grid.shape
    if x < 0 or y < 0:
        return False
    if x + pw > W or y + ph > H:
        return False
    # subgrid slice
    sub = grid[y:y+ph, x:x+pw]
    # piece fits if no cell where both sub != 0 and piece == 1
    overlap = (sub != 0) & (piece != 0)
    return not overlap.any()

def place_on(grid: np.ndarray, piece: np.ndarray, x: int, y: int, pid: int, inplace: bool = False) -> np.ndarray:
    """
    Place `piece` on a copy (or in-place if inplace=True) of `grid` with marker `pid`.
    Only modifies cells where piece == 1. Returns the modified grid.
    Raises ValueError if placement invalid (overlap or out-of-bounds).
    """
    if not fits_at(grid, piece, x, y):
        raise ValueError(f"Cannot place piece id={pid} at x={x}, y={y}: overlap or out of bounds.")
    if inplace:
        target = grid
    else:
        target = grid.copy()
    ph, pw = piece.shape
    # assign pid to cells where piece == 1
    mask = (piece != 0)
    target[y:y+ph, x:x+pw][mask] = int(pid)
    return target

def remove_from(grid: np.ndarray, piece: np.ndarray, x: int, y: int, inplace: bool = False) -> np.ndarray:
    """
    Remove a piece with known shape at (x,y) by setting those cells to zero.
    (Useful during backtracking). Does not assert pid.
    """
    ph, pw = piece.shape
    H, W = grid.shape
    if x < 0 or y < 0 or x + pw > W or y + ph > H:
        raise ValueError("Coordinates out of bounds for remove_from.")
    if inplace:
        target = grid
    else:
        target = grid.copy()
    mask = (piece != 0)
    target[y:y+ph, x:x+pw][mask] = 0
    return target

def bounding_box(grid: np.ndarray) -> Tuple[int, int]:
    """
    Return the (height, width) of the minimal bounding box covering all non-zero cells.
    If grid is empty (all zeros), returns (0, 0).
    """
    if not np.any(grid != 0):
        return 0, 0
    rows, cols = np.where(grid != 0)
    h = int(rows.max() - rows.min() + 1)
    w = int(cols.max() - cols.min() + 1)
    return h, w

def crop_to_bounding_box(grid: np.ndarray) -> np.ndarray:
    """
    Return a copy of grid cropped to its minimal bounding box (top-left origin).
    If grid is empty, returns an empty (0,0) array.
    """
    if not np.any(grid != 0):
        return np.zeros((0, 0), dtype=grid.dtype)
    rows, cols = np.where(grid != 0)
    rmin, rmax = int(rows.min()), int(rows.max()) + 1
    cmin, cmax = int(cols.min()), int(cols.max()) + 1
    return grid[rmin:rmax, cmin:cmax].copy()

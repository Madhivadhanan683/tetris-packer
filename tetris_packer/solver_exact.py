import numpy as np
from .placer import can_place, place_block, remove_block

def backtrack(grid, blocks, idx=0):
    """
    Try to place blocks recursively.
    grid: numpy 2D array
    blocks: list of block matrices (numpy arrays)
    idx: index of block being placed
    Returns grid if success, None if fail.
    """
    if idx == len(blocks):
        return grid.copy()  # success

    block = blocks[idx]
    h, w = grid.shape

    for i in range(h - block.shape[0] + 1):
        for j in range(w - block.shape[1] + 1):
            if can_place(grid, block, i, j):
                place_block(grid, block, i, j, idx+1)
                result = backtrack(grid, blocks, idx+1)
                if result is not None:
                    return result
                remove_block(grid, block, i, j)  # backtrack

    return None


def solve_exact(blocks):
    """
    Try to fit blocks into the smallest possible rectangle
    by incrementally increasing grid size.
    """
    total_cells = sum(np.sum(b) for b in blocks)

    # Start with square-ish dimensions
    min_side = int(np.sqrt(total_cells))
    for H in range(min_side, total_cells+1):
        for W in range(min_side, total_cells+1):
            if H * W < total_cells:
                continue
            grid = np.zeros((H, W), dtype=int)
            result = backtrack(grid, blocks, 0)
            if result is not None:
                aspect = max(H/W, W/H)
                return result, (H, W), aspect

    return None, None, None

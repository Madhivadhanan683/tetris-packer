import numpy as np
from tetris_packer.placer import can_place, place_block, remove_block

def test_can_place_and_place_block():
    grid = np.zeros((4, 4), dtype=int)
    block = np.array([[1, 1], [1, 0]])  # L-shape

    assert can_place(grid, block, 0, 0)
    place_block(grid, block, 0, 0, block_id=1)

    # Grid should now have ones in those positions
    expected = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    assert np.array_equal(grid, expected)

    # Removing should restore original grid
    remove_block(grid, block, 0, 0)
    assert np.array_equal(grid, np.zeros((4, 4), dtype=int))

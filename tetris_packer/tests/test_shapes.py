import numpy as np
import pytest
from tetris_packer.shapes import Shape

def test_shape_matrix_and_rotations():
    mat = np.array([[1, 1], [0, 1]])
    s = Shape(mat, name="L-block")

    # Original matrix is stored
    assert np.array_equal(s.matrix, mat)

    # Rotations should all be numpy arrays
    rots = s.rotations()
    assert len(rots) == 4
    assert all(isinstance(r, np.ndarray) for r in rots)

    # Rotating 4 times should return to original shape
    back_to_start = np.rot90(mat, -4)  # 4 clockwise rotations
    assert np.array_equal(back_to_start, mat)

import numpy as np
from tetris_packer.solver_greedy import solve_greedy
from tetris_packer.solver_exact import solve_exact
from tetris_packer.shapes import Shape

def test_solver_greedy_simple():
    # One 2x2 block
    block = Shape(np.array([[1, 1], [1, 1]]))
    grid, dims, aspect = solve_greedy([block.matrix])
    assert grid is not None
    assert dims == (2, 2)
    assert aspect == 1.0

def test_solver_exact_simple():
    # Two 1x2 bars
    block = Shape(np.array([[1, 1]]))
    blocks = [block.matrix, block.matrix]
    grid, dims, aspect = solve_exact(blocks)
    assert grid is not None
    # Should fit into 2x2 square
    assert dims == (2, 2)
    assert aspect == 1.0

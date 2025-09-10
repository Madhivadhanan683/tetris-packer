import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

TETRIS_COLORS = {
    "I-block": "cyan",
    "O-block": "yellow",
    "T-block": "purple",
    "L-block": "orange",
    "J-block": "blue",
    "S-block": "green",
    "Z-block": "red",
}

def animate_packing(placements, H, W, save_path="tetris_anim.gif"):
    """
    placements: list of tuples (block_matrix, row, col, block_name)
    H, W: grid height and width
    """
    grid = np.zeros((H, W), dtype=int)
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    ims = []

    block_id = 1
    for block, i, j, name in placements:
        color = TETRIS_COLORS.get(name, "gray")
        new_grid = grid.copy()
        h, w = block.shape
        for r in range(h):
            for c in range(w):
                if block[r, c] == 1:
                    new_grid[i+r, j+c] = block_id
        im = ax.imshow(new_grid, cmap=plt.cm.get_cmap("tab20", 20), animated=True)
        ims.append([im])
        grid = new_grid
        block_id += 1

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(save_path, writer="pillow")
    print(f"Saved animation to {save_path}")

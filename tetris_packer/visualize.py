import matplotlib.pyplot as plt
import numpy as np

def plot_grid(grid, title="Packed Grid", save_path=None):
    """
    Visualize a grid with matplotlib.
    Different block IDs get different colors.
    """
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap("tab20", np.max(grid) + 1)
    plt.imshow(grid, cmap=cmap, origin="upper")
    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

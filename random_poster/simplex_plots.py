import numpy as np
from matplotlib import pyplot as plt
from simplex import SIMPLEX_METHODS, SIMPLEX_NAMES
from plotting import scatter3D


NUM = 10000

for k, n in zip(SIMPLEX_METHODS, SIMPLEX_NAMES):
    points = np.array([SIMPLEX_METHODS[k](3) for _ in range(NUM)])
    scatter3D(points, title=f'{n} simplex', figsize=(6,6), alpha=0.4)
    plt.title(f'{n} simplex')
    plt.savefig(f'./simplex_plots/{k}.png', dpi=400)

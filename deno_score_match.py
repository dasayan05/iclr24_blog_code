from tqdm import trange

import numpy as np
import torch as th
import matplotlib.pyplot as plt

from utils import (
    GDensity,
    GMixDensity,
    EmpiricalDensity
)

g1 = GDensity([0., 0.], [
    [1., 0.9],
    [0.9, 1.]
])

g2 = GDensity([0., 0.], [
    [1., -0.9],
    [-0.9, 1.]
])

G = EmpiricalDensity(GMixDensity([g2, g1]).sample())

GRAN = 15
sigma = 0.2
N_ARROWS = 0

fig = plt.figure(figsize=(7, 7))
ax = plt.gca()
ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4])

x_grid, y_grid = np.meshgrid(np.linspace(-4, 4, GRAN), np.linspace(-4, 4, GRAN))
X = np.concatenate([x_grid.reshape((GRAN**2, 1)), y_grid.reshape((GRAN**2, 1))], -1)

i = 0
for _ in trange(200):
    ax.cla()

    G.plot_density(ax, cmap='Reds', alpha=0.5)
    # ax.scatter(G.data[:, 0], G.data[:, 1], color='red', alpha=0.2, s=1)
    
    S = G.score(X)
    S_x_grid, S_y_grid = S[:, 0, None].reshape((GRAN, GRAN)), S[:, 1, None].reshape((GRAN, GRAN))
    ax.quiver(x_grid, y_grid, S_x_grid, S_y_grid, headwidth=4, headlength=4, alpha=0.3)

    noise = np.random.randn(*G.data.shape)
    noisy_data = G.data + sigma * noise

    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4])
    fig.savefig(f'figs/test_{i}.png')
    i += 1

    G.train_score(noisy_data, noise, sigma)
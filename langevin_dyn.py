import numpy as np
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

G = GMixDensity([g2, g1])

N = GDensity([0., 0.],
    [
        [1., 0],
        [0., 1.]
    ]
)
n = EmpiricalDensity(N.sample(N=1000))

def stoch_proc_1(x, delta=0.01):
    return x + G.score(x) * delta + np.sqrt(2 * delta) * np.random.randn(*x.shape)

ITER = 50

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
XRANGE = [-3, 3]
YRANGE = [-3, 3]

for i in range(ITER):
    if i % (ITER // 50) == 0:
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        n.plot_density(ax=ax[1], cmap='Blues')
        n.plot_traj(ax=ax[0], color='black')
        G.plot_density(ax=ax[2], cmap='Reds')
        ax[0].set_xlim(XRANGE); ax[0].set_ylim(YRANGE)
        ax[1].set_xlim(XRANGE); ax[1].set_ylim(YRANGE)
        ax[2].set_xlim(XRANGE); ax[2].set_ylim(YRANGE)
        ax[0].set_title('$x_t$', fontsize=20)
        ax[1].set_title('$p_t(x)$', fontsize=20)
        ax[2].set_title('$q_{data}(x)$', fontsize=20)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.savefig(f'figs/test_{i}.png', bbox_inches='tight', pad_inches=0)
    
    n.nudge(stoch_proc_1, delta=1.e-2)

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    GDensity,
    GMixDensity,
    Uniform,
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
U = Uniform([-2, 2], [-2, 2])
n = EmpiricalDensity(N.sample(N=1000))
u = EmpiricalDensity(U.sample(N=1000))

def stoch_proc_1(x, delta=0.01):
    return x + G.score(x) * delta + np.sqrt(2 * delta) * np.random.randn(*x.shape)

ITER = 50

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
XRANGE = [-3, 3]
YRANGE = [-3, 3]

x_grid, y_grid = np.meshgrid(np.linspace(-1, 3, 10), np.linspace(-3, 1, 10))
dfdx = lambda x, y: -x / np.sqrt(x ** 2 + y ** 2)
dfdy = lambda x, y: -y / np.sqrt(x ** 2 + y ** 2)
x_vf = dfdx(x_grid, y_grid)
y_vf = dfdy(x_grid, y_grid)

p_x, p_y = 1, -2
q_x, q_y = 2, 0

for i in tqdm(range(ITER)):
    if i % (ITER // 50) == 0:
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        n.plot_density(ax=ax[0], cmap='Blues')
        ax[1].quiver(x_grid, y_grid, x_vf, y_vf)
        ax[1].scatter([p_x, ], [p_y, ])
        ax[1].scatter([q_x, ], [q_y, ], color='green')
        ax[1].scatter([0., ], [0., ], color='red')
        ax[1].text(0.3, 0.4, r'$q_{data}$', ha='center', va='center', color='red', fontsize=20)
        p_x, p_y = p_x + 5.e-2 * dfdx(p_x, p_y), p_y + 5.e-2 * dfdy(p_x, p_y)
        q_x, q_y = q_x + 5.e-2 * dfdx(q_x, q_y), q_y + 5.e-2 * dfdy(q_x, q_y)
        u.plot_density(ax=ax[2], cmap='Greens')
        ax[0].set_xlim(XRANGE); ax[0].set_ylim(YRANGE)
        ax[1].set_xlim([-1, 3]); ax[1].set_ylim([-3, 1])
        ax[2].set_xlim(XRANGE); ax[2].set_ylim(YRANGE)
        ax[0].set_title(r'$p_t\ |\ p_0 = \mathcal{N}(0, I)$', fontsize=20)
        ax[1].set_title(r'$p$ space', fontsize=20)
        ax[2].set_title(r'$p_t\ |\ p_0 = \mathcal{U}(-2, 2)$', fontsize=20)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.savefig(f'figs/test_{i}.png', bbox_inches='tight', pad_inches=0)
    
    n.nudge(stoch_proc_1, delta=5.e-3)
    u.nudge(stoch_proc_1, delta=5.e-3)

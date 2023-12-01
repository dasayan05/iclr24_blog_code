from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [1.0, 0.1],
    [1.25, 0.15],
    [1.5, 0.2],
    [1.75, 0.25],
    [2.0, 0.3],

    [-1.0, 0.1],
    [-1.25, 0.15],
    [-1.5, 0.2],
    [-1.75, 0.25],
    [-2.0, 0.3],
])

data[:, 1] *= 3

fig = plt.figure(figsize=(8, 8), dpi=600)
ax = plt.gca()

sigma = 1
ball_radius = 0.4

p1 = np.array([-1, 1])
p2 = np.array([1, 1])
p3 = np.array([0., 2.])

m1 = np.array([-3., 3.])
n_m1 = 1
m2 = np.array([2., 2.])
n_m2 = 1
m3 = np.array([-1., 2.])
n_m3 = 1


for i in trange(200):
    ax.scatter(data[:, 0], data[:, 1], color='red', marker='o')

    ax.scatter(*p1, color='blue', marker='x')
    ax.scatter(*p2, color='magenta', marker='x')
    ax.scatter(*p3, color='green', marker='x')

    ax.text(*(p1 + 0.1), r'$\tilde{x}_1$', color='blue', fontsize=15)
    ax.text(*(p2 + 0.1), r'$\tilde{x}_2$', color='magenta', fontsize=15)
    ax.text(*(p3 + 0.1), r'$\tilde{x}_3$', color='green', fontsize=15)

    noisy_data = data + sigma * np.random.randn(*data.shape)

    ax.scatter(*m1, color='blue', marker='X')
    ax.scatter(*m2, color='magenta', marker='X')
    ax.scatter(*m3, color='green', marker='X')

    ax.text(*(m1 + 0.05), r'$\mathbb{E}_{x|\tilde{x}_1}[x]$', color='blue', fontsize=15)
    ax.text(*(m2 + 0.05), r'$\mathbb{E}_{x|\tilde{x}_2}[x]$', color='magenta', fontsize=15)
    ax.text(*(m3 + 0.05), r'$\mathbb{E}_{x|\tilde{x}_3}[x]$', color='green', fontsize=15)

    ax.arrow(*p1, *((m1 - p1) * 0.5), head_width=0.07, head_length=0.07, length_includes_head=True, linewidth=2, color='gray', alpha=0.5)
    ax.arrow(*p2, *((m2 - p2) * 0.5), head_width=0.07, head_length=0.07, length_includes_head=True, linewidth=2, color='gray', alpha=0.5)
    ax.arrow(*p3, *((m3 - p3) * 0.5), head_width=0.07, head_length=0.07, length_includes_head=True, linewidth=2, color='gray', alpha=0.5)

    ax.scatter(noisy_data[:, 0], noisy_data[:, 1], color='red', marker='.', alpha=0.4)
    for clean_datum, noisy_datum in zip(data, noisy_data):
        if np.linalg.norm(noisy_datum - p1) < ball_radius:
            ax.arrow(*noisy_datum, *(clean_datum - noisy_datum),
                    head_width=0.05, head_length=0.05, length_includes_head=True, linewidth=1, color='blue', alpha=0.3,
                    linestyle='--')
            m1 = (m1 * n_m1 + clean_datum) / (n_m1 + 1)
            n_m1 += 1
        if np.linalg.norm(noisy_datum - p2) < ball_radius:
            ax.arrow(*noisy_datum, *(clean_datum - noisy_datum),
                    head_width=0.05, head_length=0.05, length_includes_head=True, linewidth=1, color='magenta', alpha=0.3,
                    linestyle='--')
            m2 = (m2 * n_m2 + clean_datum) / (n_m2 + 1)
            n_m2 += 1
        if np.linalg.norm(noisy_datum - p3) < (ball_radius * 2):
            ax.arrow(*noisy_datum, *(clean_datum - noisy_datum),
                    head_width=0.05, head_length=0.05, length_includes_head=True, linewidth=1, color='green', alpha=0.3,
                    linestyle='--')
            m3 = (m3 * n_m3 + clean_datum) / (n_m3 + 1)
            n_m3 += 1

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-1, 2.5])

    ax.axis('off')

    # plt.draw()
    # plt.pause(0.1)

    plt.savefig(f'figs/test_{i}.png', bbox_inches='tight', pad_inches=0)
    ax.cla()
import typing as ty
import torch as th
import numpy as np
from scipy.stats import multivariate_normal
import seaborn as sns
import queue

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis = axis)
    return np.take_along_axis(a, idx, axis = axis)

class GDensity:

    def __init__(self, mean, cov) -> None:
        super().__init__()
        self.mean, self.cov = np.array(mean), np.array(cov)
        self.dist = multivariate_normal(self.mean, self.cov)
        self.data = self.sample()
    
    def sample(self, N=1000):
        return self.dist.rvs(size=N)

    def likelihood(self, x):
        lpdf = self.dist.logpdf(x)[:, None]
        lpdf[lpdf < -20] = -20 # just for numerical stability
        return np.exp(lpdf)
    
    def score(self, x, normalize = False):
        if len(x.shape) == 1:
            x = x[None, :]
        
        score = - np.einsum('ab,Nb->Na', np.linalg.inv(self.cov), x - self.mean[None, :])
        if normalize:
            score = score / np.linalg.norm(score, ord=0, axis=1, keepdims=True)
        return score.squeeze()

    def plot_density(self, ax, cmap='Reds'):
        EmpiricalDensity.plot_density(self, ax, cmap=cmap)

class Uniform:

    def __init__(self, xrange, yrange) -> None:
        self.xlow, self.xhigh = xrange
        self.xrange = self.xhigh - self.xlow
        self.ylow, self.yhigh = yrange
        self.yrange = self.yhigh - self.ylow
        self.data = self.sample()
    
    def sample(self, N=1000):
        x = np.random.rand(N, 1) * self.xrange + self.xlow
        y = np.random.rand(N, 1) * self.yrange + self.ylow
        return np.concatenate([x, y], axis=-1)

class GMixDensity:
    
    def __init__(self, components: ty.List[GDensity]) -> None:
        self.components = components
        self.M = len(self.components)
        self.data = self.sample()
    
    def sample(self, N=1000):
        NS = [N // self.M for _ in range(self.M)]
        extra = N - sum(NS)
        NS[-1] += extra

        samples = []
        for m in range(self.M):
            s = self.components[m].sample(NS[m])
            samples.append(s)
        
        return np.concatenate(samples, 0)

    def score(self, x, normalize = False):
        numerator = sum([comp.score(x, normalize=normalize) * comp.likelihood(x) for comp in self.components])
        denominator = sum([comp.likelihood(x) for comp in self.components])
        score = numerator / denominator
        return score

    def plot_density(self, ax, cmap='Reds'):
        GDensity.plot_density(self, ax, cmap=cmap)


class EmpiricalDensity:

    N_TRAJ = 70
    TRAJ_LEN = 7

    def __init__(self, data) -> None:
        assert len(data.shape) == 2 and data.shape[1] == 2, "data array not properly sized"
        self._data = queue.Queue(EmpiricalDensity.TRAJ_LEN)
        self._data.put(data)

        self.score_model = th.nn.Sequential(
            th.nn.Linear(2, 12),
            th.nn.Tanh(),
            th.nn.Linear(12, 2)
        ).double()

        self.opt = th.optim.Adam(self.score_model.parameters(), lr=5e-3)
    
    @property
    def data(self):
        return self._data.queue[-1]

    @data.setter
    def data(self, d):
        if self._data.full():
            self._data.get()

        self._data.put(d)
    
    def plot_density(self, ax, cmap='Reds', **kwargs):
        sns.kdeplot(x=self.data[:, 0], y=self.data[:, 1], ax=ax, cmap=cmap, fill=True, **kwargs)
    
    def plot_traj(self, ax, color='black'):
        if self._data.qsize() > 1:
            traj = np.stack(list(self._data.queue), 1)
            for i in range(EmpiricalDensity.N_TRAJ):
                selected_traj = traj[i]
                ax.plot(selected_traj[:, 0], selected_traj[:, 1], color=color, linewidth=0.5, alpha=0.2)
        
        latest = self._data.queue[-1]
        selected_latest = latest[:EmpiricalDensity.N_TRAJ, ...]
        ax.scatter(selected_latest[:, 0], selected_latest[:, 1], color=color, s=2)
    
    def nudge(self, fn, delta = 0.01):
        self.data = fn(self.data, delta)

    def score(self, x, normalize=False):
        with th.no_grad():
            return self.score_model(th.from_numpy(x)).numpy()
    
    def train_score(self, noisy_data, target, sigma=0.1):
        noisy_data = th.from_numpy(noisy_data)
        target = th.from_numpy(target)
        loss = self.score_model(noisy_data) - (- target / sigma)
        self.opt.zero_grad()
        loss = (th.linalg.vector_norm(loss, ord=2, dim=-1) ** 2).mean()
        loss.backward()
        self.opt.step()
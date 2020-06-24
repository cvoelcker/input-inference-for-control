from abc import ABC, abstractmethod

import jax.numpy as np
from jax.random import PRNGKey, gumbel, split
from jax import vmap

import numpy as onp

import torch
from torch.distributions import Gumbel

from pi2c.utils import to_polar, to_polar_torch

class EnvCostFunction(ABC):
    normalized = False
    constant_goal = False
    torch = True

    @abstractmethod
    def _cost(self, x, u, xg, ug):
        pass
    
    def cost(self, x, u, xg=None, ug=None):
        if self.constant_goal:
            return self._cost(x, u, self.xg, self.ug)
        else:
            return self._cost(x, u, xg, ug)

    def __call__(self, x, u, xg=None, ug=None):
        if self.torch:
            return self.cost(x, u, xg, ug)
        else:
            x = np.concatenate((x, u), 1)
            return self.cost_jax_(x)


class QRCost(EnvCostFunction):
    normalized = False
    constant_goal = False
    Q = None
    R = None

    def __init__(self, Q, R):
        self.Q = torch.Tensor(Q)
        self.R = torch.Tensor(R)

    def _cost(self, x, u, xg, ug):
        _x = x - xg
        _u = u - ug

        Q_cost = (_x.T).dot(Q).dot(_x)
        R_cost = (_u.T).dot(R).dot(_u)

        return Q_cost + R_cost


class StaticQRCost(QRCost):
    normalized = False
    constant_goal = True
    Q = None
    R = None

    def __init__(self, Q, R, xg, ug):
        super().__init__(Q, R)
        self.zg = torch.Tensor(onp.concatenate([xg, ug], 1))
        self.dim = self.zg.size()[1]
        self.QR = torch.zeros((self.dim, self.dim))
        self.QR[:xg.size,:xg.size] = torch.Tensor(Q)
        self.QR[xg.size:,xg.size:] = torch.Tensor(R)
        self.xg = torch.Tensor(xg)
        self.ug = torch.Tensor(ug)

    def _cost(self, x, u, xg, ug):
        _x = x - xg
        _u = u - ug

        Q_cost = torch.diag(_x @ self.Q @ _x.T)
        R_cost = torch.diag(_u @ self.R @ _u.T)
        return -(Q_cost + R_cost)

    def cost_jax(self, x):
        """Replicates the cost function to enable jax differentiation
        jax maps vectorized functions explicitely, so the cost function
        assumes a single vector x

        Args:
            x (np.array): the concatenated state control vector

        Returns:
            float: quadratic cost
        """
        x = x.reshape(1, -1)
        return -(x @ self.QR.numpy() @ x.T)[0,0]

    def cost_jax_(self, x):
        """Replicates the cost function to enable jax differentiation
        jax maps vectorized functions explicitely, so the cost function
        assumes a single vector x

        Args:
            x (np.array): the concatenated state control vector

        Returns:
            float: quadratic cost
        """
        return -np.diag(x @ self.QR.numpy() @ x.T)


class PendulumCost(StaticQRCost):

    def _cost(self, x, u, xg, ug):
        _x = to_polar_torch(x)
        _x = _x - xg
        _u = u - ug

        Q_cost = torch.diag(_x @ self.Q @ _x.T)
        R_cost = torch.diag(_u @ self.R @ _u.T)
        return -(Q_cost + R_cost)

    def cost_jax(self, x):
        """Replicates the cost function to enable jax differentiation
        jax maps vectorized functions explicitely, so the cost function
        assumes a single vector x

        Args:
            x (np.array): the concatenated state control vector

        Returns:
            float: quadratic cost
        """
        u = x[..., -1:].reshape(1, -1)
        x = x[..., :-1].reshape(1, -1)
        x = to_polar(x.T).T
        x = np.concatenate([x,u], -1)
        return -(x @ self.QR.numpy() @ x.T)[0,0]

    def cost_jax_(self, x):
        """Replicates the cost function to enable jax differentiation
        jax maps vectorized functions explicitely, so the cost function
        assumes a single vector x

        Args:
            x (np.array): the concatenated state control vector

        Returns:
            float: quadratic cost
        """
        u = x[..., -1:]
        x = x[..., :-1]
        x = to_polar(x.T).T
        return -np.diag(x @ self.Q.numpy() @ x.T + u @ self.R.numpy() @ u.T)


class Cost2Prob():
    def __init__(self, cost, backend='torch'):
        self.normalized = cost.normalized
        self.constant_goal = cost.constant_goal
        self.c = cost
        self._key = PRNGKey(0)

    @property
    def key(self):
        self._key, sk = split(self._key)
        return sk

    def likelihood(self, x, u, alpha=1., xg=None, ug=None):
        return np.exp(alpha * self.c.cost(x, u, xg, ug))

    def log_sample(self, x, u, n, alpha=1., xg=None, ug=None):
        c = self.c.cost(x, u, xg, ug)
        costs = c.reshape(-1,1) # unnormalized log probabilities
        samples = Gumbel(loc=0., scale=1.).sample((x.shape[0],n))
        log_gumbel = alpha * costs + samples
        _, choices = torch.max(log_gumbel, 0)
        return choices, alpha*costs

    def cost_jax(self, x):
        return self.c.cost_jax_(x)

    def log_sample_jax(self, x, u, n, alpha=1, xg=None, ug=None):
        x = np.concatenate((x,u), 1)
        c = vmap(self.c.cost_jax)(x)
        costs = c.reshape(-1,1)
        samples = gumbel(self.key, (costs.shape[0], n))
        choices = np.argmax(samples + alpha * costs, 0)
        return choices, alpha * costs
    
    def __call__(self, x, u, alpha=1., xg=None, ug=None):
        return self.likelihood(x, u, alpha, xg, ug)

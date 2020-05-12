import jax.numpy as np
from jax import random, jit
from abc import ABC, abstractmethod

class EnvCostFunction(ABC):
    normalized = False
    constant_goal = False

    @abstractmethod
    def _cost(self, x, u, xg, ug):
        pass
    
    def cost(self, x, u, xg=None, ug=None):
        if self.constant_goal:
            return self._cost(x, u, self.xg, self.ug)
        else:
            return self._cost(x, u, xg, ug)

    def __call__(self, x, u, xg=None, ug=None):
        return self.cost(x, u, xg, ug)


class QRCost(EnvCostFunction):
    normalized = False
    constant_goal = False
    Q = None
    R = None

    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

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
        self.xg = xg
        self.ug = ug
        self.zg = np.concatenate([xg, ug], 1)
        self.dim = self.zg.size

        # construct QR
        _Qupper = np.concatenate([Q, np.zeros((Q.shape[0], R.shape[0]))], axis=1)
        _Rlower = np.concatenate([np.zeros((R.shape[0], Q.shape[0])), R], axis=1)
        self.QR = np.concatenate([_Qupper, _Rlower], axis=0)

    def _cost(self, x, u, xg, ug):
        _x = x - xg
        _u = u - ug
        
        Q_cost = np.diag(_x @ self.Q @ _x.T)
        R_cost = np.diag(_u @ self.R @ _u.T)

        return -(Q_cost + R_cost)


def _log_sample(x, u, alpha, n, xn, Q, R, key):
    costs = -alpha * (np.diag(x @ Q @ x.T) + np.diag(u @ R @ u.T))
    samples = random.gumbel(key, (xn, n))
    log_gumble = np.expand_dims(costs,1) + samples
    choices = np.argmax(log_gumble, 0)
    return choices, costs


class Cost2Prob():
    def __init__(self, cost, n_x, n_samples):
        self.normalized = cost.normalized
        self.constant_goal = cost.constant_goal
        self.c = cost
        self.key = random.PRNGKey(0)

        self.log_sample = lambda x, u, alpha: _log_sample(x, u, alpha, n_samples, n_x, self.c.Q, self.c.R, self.key)


    def likelihood(self, x, u, alpha=1., xg=None, ug=None):
        return np.exp(alpha * self.c.cost(x, u, xg, ug))

    def __call__(self, x, u, alpha=1., xg=None, ug=None):
        return self.likelihood(x, u, alpha, xg, ug)

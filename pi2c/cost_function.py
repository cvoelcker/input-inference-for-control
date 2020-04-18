import numpy as np
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
        self.QR = np.zeros((self.dim, self.dim))
        self.QR[:xg.size,:xg.size] = Q
        self.QR[xg.size:,xg.size:] = R

    def _cost(self, x, u, xg, ug):
        _x = x - xg
        _u = u - ug

        Q_cost = np.diag((_x).dot(self.Q).dot(_x.T))
        R_cost = np.diag((_u).dot(self.R).dot(_u.T))

        return -(Q_cost + R_cost)


class Cost2Prob():
    def __init__(self, cost):
        self.normalized = cost.normalized
        self.constant_goal = cost.constant_goal
        self.c = cost

    def likelihood(self, x, u, alpha=1., xg=None, ug=None):
        return np.exp(alpha * self.c.cost(x, u, xg, ug))
    
    def __call__(self, x, u, alpha=1., xg=None, ug=None):
        return self.likelihood(x, u, alpha, xg, ug)

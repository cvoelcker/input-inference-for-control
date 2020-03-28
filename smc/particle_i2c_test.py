import numpy as np
from code import interact

from pi2c.env import make_env
from pi2c.model import make_env_model
from pi2c.cost_function import QRCost, StaticQRCost, Cost2Prob
from pi2c.particle_i2c import ParticleI2cGraph

import matplotlib.pyplot as plt


class Experiment:
    '''Class for keeping track of an experiment'''
    ENVIRONMENT: str = None
    N_DURATION: int = None


def build_q():
    Q = np.eye(2) * 10.
    R = np.eye(1) * 1.
    x = np.array([[0., 0.]])
    u = np.array([[0.]])
    cost = StaticQRCost(Q, R, x, u)
    prob = Cost2Prob(cost)
    return cost, prob


def build_sys(n):
    exp = Experiment()
    exp.ENVIRONMENT = 'LinearDisturbed'
    exp.N_DURATION = n
    env = make_env(exp)
    return env


if __name__ == "__main__":
    print('Hello')
    cost, prob = build_q()
    sys = build_sys(100)
    sys.init_env()
    
    # .... somewhere in your code
    # interact(local=locals())
    
    
    particle_graph = ParticleI2cGraph(
        sys, cost, 100, 10000, 1000, np.array([5., 5.]), 0.00001, np.array([0.]), 100000., 100., 10.)
    for i in range(100):
        particle_graph._expectation(0.0001)
        print('Updated graph {}'.format(i))
        # print(particle_graph.cells[0].back_particles)

        sys.init_env()
        costs = []
        costs_y = []
        x = sys.x0.copy()
        y = sys.x0.copy()
        for i in range(100):
            u = particle_graph.get_policy(x.reshape(-1), i)
            u0 = np.array([[0.]])
            costs.append(cost(x.reshape(-1), u))
            costs_y.append(cost(y.reshape(-1), u0))
            x = sys.sample(x, u)
            y = sys.sample(y, u0)
        plt.plot(costs)
        plt.plot(costs_y)
        plt.show()

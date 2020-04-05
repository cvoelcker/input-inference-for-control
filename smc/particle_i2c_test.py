import sys

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
    num_p = int(sys.argv[1])
    print('Hello')
    cost, prob = build_q()
    sys = build_sys(100)
    sys.init_env()
    
    # .... somewhere in your code
    # interact(local=locals())
    
    
    particle_graph = ParticleI2cGraph(
        sys, cost, 100, num_p, num_p//10, np.array([5., 5.]), 0.00001, np.array([0.]), 100000., 100., 100.)
    for i in range(100):
        alpha = 0.0001
        alpha = particle_graph.run(alpha, False)
        for c in particle_graph.cells:
            c.update_u_prior()
        
        
        print('Updated graph {}'.format(i))
        # # print(particle_graph.cells[0].back_particles)

        # sys.init_env()
        # costs = []
        # traj_1 = []
        # contr_1 = []
        # costs_y = []
        # traj_2 = []
        # contr_2 = []
        # x = sys.x0.copy()
        # y = sys.x0.copy()
        # for i in range(100):
        #     u = particle_graph.get_policy(x.reshape(-1), i).reshape(-1, 1)
        #     u0 = np.array([[0.]])
        #     if np.isnan(u):
        #         u = u0
        #     contr_1.append(u)
        #     traj_1.append(x)
        #     contr_2.append(u0)
        #     traj_2.append(y)
        #     costs.append(cost(x.reshape(-1), u))
        #     costs_y.append(cost(y.reshape(-1), u0))
        #     x = sys.sample(x, u)
        #     y = sys.sample(y, u0)
        # print(np.mean(costs))
        # print(np.mean(costs_y))
        # np.save('test_results/cost_{}_{}.npy'.format(num_p, i), costs)
        # np.save('test_results/costY_{}_{}.npy'.format(num_p, i), costs_y)
        # np.save('test_results/contr1_{}_{}.npy'.format(num_p, i), contr_1)
        # np.save('test_results/contr2_{}_{}.npy'.format(num_p, i), contr_2)
        # np.save('test_results/traj1_{}_{}.npy'.format(num_p, i), traj_1)
        # np.save('test_results/traj2_{}_{}.npy'.format(num_p, i), traj_2)

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

def eval_controller(n, controller, system, cost):
    costs = 0.
    x = system.init_env()
    for i in range(n):
        u = controller.get_policy(x.flatten(), i).reshape(-1,1)
        x = system.forward(u)
        costs += cost(x.flatten(), u)
    return costs/n
    

if __name__ == "__main__":
    num_p = int(sys.argv[1])
    u_samples = int(sys.argv[2])
    num_runs = int(sys.argv[3])
    print('Hello')
    cost, prob = build_q()
    sys = build_sys(100)
    
    alpha = 1e-5
    particle_graph = ParticleI2cGraph(
        sys, cost, 100, num_p, num_p//10, np.array([5., 5.]), 0.1, np.array([0., 0., 0.]), 10000., alpha, 1, u_samples, num_runs)

    costs_over_run = []
    alpha_over_run = []
    sys.init_env()
    alpha = particle_graph.run(alpha, False, 1)
    for i in range(100):
        costs = eval_controller(100, particle_graph, sys, cost)
        print(costs)
        # costs_over_run.append(costs)
        sys.init_env()
        # alpha_over_run.append(alpha)
        particle_graph.run(alpha, False, 1)
        
        print('Updated graph {}'.format(i))

    np.savetxt('results/cost_ia{}_np{}_nu{}_nr{}.npy'.format(alpha_over_run[-1], num_p, u_samples, num_runs), costs_over_run)

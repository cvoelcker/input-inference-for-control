import sys
from collections import namedtuple

import numpy as np

from pi2c.env import make_env
from pi2c.model import make_env_model
from pi2c.cost_function import QRCost, StaticQRCost, Cost2Prob
from pi2c.particle_i2c import ParticleI2cGraph, ParticlePlotter
from pi2c.utils import get_particle_i2c_config


def build_quadratic_q(dim_x, dim_u, q=None, r=None):
    Q = np.eye(dim_x)
    R = np.eye(dim_u)
    if Q is None: Q *= 10.
    else: Q *= q
    if R is None: R *= 1.
    else: R *= r

    # dynamic trajectory needs to be implemented here
    x = np.array([[0., 0.]])
    u = np.array([[0.]])
    cost = StaticQRCost(Q, R, x, u)
    prob = Cost2Prob(cost)
    return cost, prob


def build_env(env, horizon):
    EXP = namedtuple('Experiment', ['ENVIRONMENT', 'N_DURATION'])
    exp = EXP(env, horizon)
    env = make_env(exp)
    return env


def build_particle_graph(graph_config):
    graph = ParticleI2cGraph()
    return None


def build_experiment(config):
    env = build_env(config.ENVIRONMENT.env_name, config.ENVIRONMENT.horizon)
    cost, _ = build_quadratic_q(env.dim_x, env.dim_u, config.ENVIRONMENT.cost.Q, config.ENVIRONMENT.cost.R)
    graph = build_particle_graph(config.GRAPH)
    graph.set_policy(config.POLICY)
    graph.set_update(config.OPTIMIZER)
    graph.set_logging(config.LOGGING)


if __name__ == "__main__":
    config = get_particle_i2c_config(sys.argv[1:], 'config/particle_i2c.yml')
    graph, env, cost = build_experiment(config)
    print(config)

    # num_p = int(sys.argv[1])
    # u_samples = int(sys.argv[2])
    # num_runs = int(sys.argv[3])
    # log_dir = sys.argv[4]
    # print('Hello')
    # cost, prob = build_q()
    # sys = build_sys(100)
    # 
    # alpha = 1e-5
    # particle_graph = ParticleI2cGraph(
    #     sys, cost, 100, num_p, num_p//10, np.array([0., 0.]), 0.0001, np.array([0., 0., 0.]), 100., alpha, 2, u_samples, num_runs)
    # plotter = ParticlePlotter(particle_graph)

    # costs_over_run = []
    # alpha_over_run = []
    # sys.init_env()
    # # alpha = particle_graph.run(alpha, False, 1)
    # for i in range(1000):
    #     sys.init_env()
    #     alpha = particle_graph.run(alpha, i + 1, False, 2, log_dir)
    #     
    #     print('Updated graph {}, new alpha {}'.format(i, alpha))

    # np.savetxt('results/cost_ia{}_np{}_nu{}_nr{}.npy'.format(alpha_over_run[-1], num_p, u_samples, num_runs), costs_over_run)

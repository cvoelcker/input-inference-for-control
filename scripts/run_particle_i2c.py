import sys
import os
from collections import namedtuple

from tqdm import tqdm
import numpy as np

from pi2c.env import make_env
from pi2c.model import make_env_model
from pi2c.cost_function import QRCost, StaticQRCost, Cost2Prob
import pi2c.particle_i2c
from pi2c.particle_i2c import ParticleI2cGraph#, ParticlePlotter
from pi2c.utils import get_particle_i2c_config
from pi2c.particle_visualization import ParticlePlotter


def build_quadratic_q(dim_x, dim_u, q=None, r=None):
    Q = np.eye(dim_x)
    R = np.eye(dim_u)
    if q is None: 
        Q *= 10.
    else:
        q = np.array(q)
        print(q)
        Q *= q
    if r is None:
        R *= 1.
    else:
        r = np.array(r)
        R *= r

    # dynamic trajectory needs to be implemented here
    x = np.zeros_like(q).reshape(1, -1)
    u = np.array([[0.]])
    cost = StaticQRCost(Q, R, x, u)
    prob = Cost2Prob(cost)
    return cost, prob


def build_env(env, horizon):
    EXP = namedtuple('Experiment', ['ENVIRONMENT', 'N_DURATION'])
    exp = EXP(env, horizon)
    env = make_env(exp)
    return env


def build_particle_graph(config):
    T = config.ENVIRONMENT.horizon
    num_p = config.GRAPH.num_particles
    num_u_samples = config.GRAPH.num_policy_samples
    M = config.GRAPH.num_backwards
    mu_x0 = np.array(config.ENVIRONMENT.init_state_mean)
    sig_x0 = np.array(config.ENVIRONMENT.init_state_var)
    alpha_init = config.GRAPH.init_alpha
    graph = ParticleI2cGraph(T, num_p, num_u_samples, M, mu_x0, sig_x0, alpha_init)
    return graph


def build_experiment(config):
    env = build_env(config.ENVIRONMENT.env_name, config.ENVIRONMENT.horizon)
    print(env)
    print(env.dim_x)
    cost, _ = build_quadratic_q(env.dim_x, env.dim_u, config.ENVIRONMENT.cost.Q, config.ENVIRONMENT.cost.R)
    graph = build_particle_graph(config)
    graph.set_env(env, cost)
    graph.set_policy(config.POLICY.type, config.POLICY.smoothing, config.POLICY)
    if config.POLICY.type == 'VSMC':
        pi2c.particle_i2c.BACKEND = 'torch'
        graph.set_optimizer('gradient', config.OPTIMIZER.batch_size, config.OPTIMIZER.gradient_norm, config.OPTIMIZER.lr)
    elif config.POLICY.type == 'mixture':
        pi2c.particle_i2c.BACKEND = 'jax'
        graph.set_optimizer('em', config.OPTIMIZER.batch_size)
    return graph, env, cost


def build_logger(graph, config):
    logger = ParticlePlotter(graph, config)
    return logger


if __name__ == "__main__":
    config = get_particle_i2c_config(sys.argv[1:], 'config/particle_i2c.yml')
    log_dir = 'logging/' + config.LOGGING.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    graph, env, cost = build_experiment(config)
    logger = build_logger(graph, config)

    steps_per_log = config.LOGGING.log_every
    epochs = config.LOGGING.em_steps // steps_per_log

    for i in tqdm(range(epochs)):
        env.init_env()
        alpha, loss, forward_particles, weights, backward_particles = graph.run(i, steps_per_log, log_dir, config.ENVIRONMENT.init_state_bimodal)
        logger.plot_all(
            alpha, 
            forward_particles, 
            backward_particles, 
            weights, 
            config.ENVIRONMENT.env_name, 
            env, 
            cost, 
            repeats=10, 
            random_starts=True)

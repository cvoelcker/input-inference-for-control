from contextlib import contextmanager
import datetime
from distutils.spawn import find_executable
import jax.numpy as np
from jax import jit
import torch
import os
from sys import argv
import time
from config_parser import config_parser

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal


def finite_horizon_lqr(H, A, a, B, Q, R, x0, xg, ug, dim_x, dim_u):
    a = a.squeeze()
    xg = xg.squeeze()

    K = np.zeros((H, dim_u, dim_x))
    k = np.zeros((H, dim_u))
    Ps = np.zeros((H, dim_x, dim_x))
    ps = np.zeros((H, dim_x))
    P = Q
    p = -Q @ xg
    for i in range(H-1, -1, -1):
        Ps[i, :, :] = P.squeeze()
        ps[i, :] = p.squeeze()
        _M = R + B.T @ P @ B
        _Minv = np.linalg.inv(_M)
        _K = np.linalg.inv(_M) @ B.T
        K[i, :, :] = -_Minv @ B.T @ P @ A
        k[i, :] = -_Minv @ (B.T @ P @ a + B.T @ p - R @ ug)
        _P = Q + A.T @ P @ A - A.T @ P @ B @ _Minv @ B.T @ P @ A
        p = A.T @ (P @ a + p - P @ B @ _Minv @ ( B.T @ (P @ a + p) - R @ ug )) - Q @ xg
        P = _P

    x_lqr = np.zeros((H, dim_x))
    u_lqr = np.zeros((H, dim_u))
    u_K = np.zeros((H, dim_u))
    u_k = np.zeros((H, dim_u))
    x = x0
    cost = 0.
    for i in range(H):
        x_lqr[i, :2] = x.squeeze()
        u_K[i] = K[i, :, :].dot(x).squeeze()
        u_k[i] = k[i, :].squeeze()
        u_lqr[i] = u_K[i] + u_k[i]
        u = u_lqr[i]
        c = x.T.dot(Q.dot(x)) + u.T.dot(R.dot(u))
        cost += c
        x = A.dot(x).squeeze() + B.dot(u_lqr[i]) + a
    cost += x.T.dot(Q.dot(x))

    return x_lqr, u_lqr, K, k, cost[0,0], Ps, ps

class TrajectoryData(object):

    def __init__(self, x_perturbation_noise, y_perturbation_noise, n_aug=1):
        self.x_exp = []
        self.y_exp = []
        self.x_noise = x_perturbation_noise
        self.y_noise = y_perturbation_noise
        self.n_aug = n_aug

    def add(self, x, y):
        self.x_exp.append(x)
        self.y_exp.append(y)
        if self.n_aug > 0:
            for _ in range(self.n_aug):
                self.x_exp.append(x + np.random.randn(*x.shape).dot(self.x_noise))
                self.y_exp.append(y + np.random.randn(*y.shape).dot(self.y_noise))

        _x = np.vstack(self.x_exp)
        _y = np.vstack(self.y_exp)
        return _x, _y

class TrajectoryEvaluator(object):

    def __init__(self, horizon, W, sg):
        self.horizon = horizon
        self.W = W
        self.sg = sg.reshape(-1, 1)
        self.actual_cost = []
        self.planned_cost = []

        d = W.shape[0]
        assert self.W.shape[1] == d
        assert self.sg.shape[0] == d


    def dist(self, y):
        err = y.reshape(-1, 1) - self.sg
        return (err.T @ self.W @ err)[0,0]

    def _eval_traj(self, y):
        return sum([self.dist(y[i,:])
                    for i in range(self.horizon+1)])

    def eval(self, actual_traj, planned_traj):
        self.actual_cost.append(
            self._eval_traj(actual_traj))
        self.planned_cost.append(
            self._eval_traj(planned_traj))

    def plot(self, name, res_dir=None):
        f = plt.figure()
        plt.title("Trajectory Cost over evaluations ")
        plt.plot(self.actual_cost, 'ro-', label="Actual")
        plt.plot(self.planned_cost, 'bo-', label="Planned")
        plt.legend()
        plt.xlabel("Evaluations")
        plt.ylabel("Cost")
        if res_dir is not None:
            plt.savefig(os.path.join(res_dir,
                "traj_eval_{}.png".format(name)),
                bbox_inches='tight', format='png')
            plt.close(f)

    def save(self, name, res_dir):
        actual = np.asarray(self.actual_cost)
        plan = np.asarray(self.planned_cost)
        np.save(os.path.join(res_dir, "cost_actual_{}.npy".format(name)), actual)
        np.save(os.path.join(res_dir, "cost_plan_{}.npy".format(name)), plan)



def converged_list(data, tol):
    if len(data) > 2:
        return (abs(data[-1] - data[-2]) / abs(data[-2])) < tol
    else:
        return False

@contextmanager
def profile(name, log):
    t = time.time()
    yield
    tt = int(time.time() - t)
    if log:
        print("{} took {:d}m {:d}s".format(name, tt // 60, tt % 60))

DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_results_folder(config, seed, name):
    folder = "{}_{}_{}_{}".format(DATETIME, config, seed, name.replace(" ", "_"))
    res_dir = os.path.join("_results", folder)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    return res_dir


def configure_plots():
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['figure.figsize'] = [19, 19]
    matplotlib.rcParams['legend.fontsize'] = 16
    matplotlib.rcParams['axes.titlesize'] = 22
    matplotlib.rcParams['axes.labelsize'] = 22
    if find_executable("latex"):
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.unicode'] = True


@jit
def to_polar(x):
    x0 = np.arctan2(x[0, :], x[1, :])
    x1 = x[2, :]
    return np.stack([x0, x1], 0)


@jit
def to_euclidean(x):
    x0 = np.sin(x[0, :])
    x1 = np.cos(x[0, :])
    x2 = x[1, :]
    return np.stack([x0, x1, x2], 0)


def to_polar_torch(x):
    x0 = torch.atan2(x[:, 0], x[:, 1])
    x1 = x[:, 2]
    return np.stack([x0, x1], 1)


def to_euclidean_torch(x):
    x0 = torch.sin(x[:, 0])
    x1 = torch.cos(x[:, 0])
    x2 = x[:, 1]
    return torch.stack([x0, x1, x2], 1)


# particle experiments argparser
def get_particle_i2c_config(args, config_file):
    parser = config_parser.ConfigGenerator(config_file)
    parsed = parser(args)
    return parsed


class GaussianPrior():

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dim = len(mu)

        self.pdf = multivariate_normal(mu, sigma)

    def likelihood(self, x):
        return self.pdf.pdf(x)

    def sample(self, n):
        samples = []
        for _ in range(n):
            samples.append(self.pdf.rvs())
        return np.array(samples)

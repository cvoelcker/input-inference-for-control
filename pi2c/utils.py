from contextlib import contextmanager
import datetime
from distutils.spawn import find_executable
import numpy as np
import os
import time
from abc import ABC, abstractmethod

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
        return err.T.dot(self.W.dot(err))[0,0]

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


class Distribution(ABC):

    def __call__(self, x):
        return self.likelihood(x)

    @abstractmethod
    def likelihood(self, x):
        pass

    @abstractmethod
    def sample(self, n):
        pass


class GMM(Distribution):
    """A Gaussian Mixture Model with constant sigma for each mixture component,
    to model a kernel density estimate
    """

    def __init__(self, mu, pi, sig):
        """[summary]
        
        Arguments:
            mu {np.ndarray} -- means of the mixture components
            pi {np.ndarray} -- weights of the mixture components
            sig {float} -- covariance of each mixture comp: np.eye() * sig
        """
        self.dim = mu.shape[1]
        self.sig_scale = sig
        self.sig = np.eye(self.dim) * sig
        self.mu = mu
        self.num_components = len(pi)

        self.gaussians = []
        
        for i in range(mu.shape[0]):
            pdf = multivariate_normal(mu[i], sig)
            self.gaussians.append(pdf)
        
        self.pi = pi

    def likelihood(self, x):
        """COmputes the likelihood under the gmm using a weighted sum
        
        Arguments:
            x {[type]} -- [description]
        """
        prob = 0.
        for g, p in zip(self.gaussians, self.pi):
            prob += p * g.pdf(x)
        return prob

    def sample(self, n):
        """Implements sampling from the GMM
        
        Arguments:
            n {int} -- number of samples to draw
        """
        samples = []
        comp = np.random.choice(
            np.arange(self.num_components), 
            size=n, 
            replace=True, 
            p=self.pi)
        for c in comp:
            samples.append(self.gaussians[c].rvs())
        return np.array(samples)

    def mean(self):
        pi = self.pi.reshape(self.num_components, 1)
        return np.sum(self.mu * pi, 0)

    def marginalize(self, idx):
        """Marginalizes a continuous range of the GMM domain
        and returns a GMM over that range
        
        Arguments:
            idx {np.index_array} -- indices of the remaining observations
        
        Returns:
            GMM -- A new mixture model
        """
        new_mu = self.mu[:, idx]
        return GMM(new_mu, self.pi, self.sig_scale)

    def condition(self, x, idx):
        """Builds a new GMM which is conditioned on an observation

        Arguments:
            x {np.array} -- observation to condition on
            idx {np.index_array} -- index array of the observation
        """
        obs_mask = np.zeros(self.dim, dtype=np.bool_)
        obs_mask[idx] = True
        var_mask = np.where(~obs_mask)[0]
        
        gmm_obs = self.marginalize(idx)
        gmm_var = self.marginalize(var_mask)
        
        reweight = np.array([g.pdf(x) for g in gmm_obs.gaussians])
        reweight = gmm_var.pi * reweight
        reweight /= np.sum(reweight)
        gmm_var.pi = reweight

        return gmm_var

    def conditional_mean(self, x, idx):
        return self.condition(x, idx).mean()

    def conditional_sample(self, x, idx):
        return self.condition(x, idx).sample()


class GaussianPrior(Distribution):

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

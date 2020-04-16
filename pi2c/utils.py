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
        self.num_components = len(pi)
        self.dim = mu.shape[1]
        self.var_scale = sig
        self.var = np.eye(self.dim) * sig
        self.var = np.tile(self.var, (self.num_components, 1, 1))
        self.mu = mu
        self.pi = pi
        self.init_pdf()

    def init_pdf(self):
        self.gaussians = []
        self.sig = []
        for i in range(self.num_components):
            self.sig.append(np.linalg.cholesky(self.var[i]))
            pdf = multivariate_normal(self.mu[i], self.var[i])
            self.gaussians.append(pdf)#
        self.sig = np.array(self.sig)

    def likelihood(self, x):
        """Computes the likelihood under the gmm using a weighted sum
        
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
        comp = np.random.choice(
            np.arange(self.num_components), 
            size=n, 
            replace=True, 
            p=self.pi)
        
        ran = np.random.randn(n, 1, self.dim)
        # print((self.mu[comp]).shape)
        # print(ran.shape)
        # print(self.sig[comp].shape)

        samples = self.mu[comp].reshape(n, 1, -1) + ran @ self.sig[comp]

        return samples.reshape(n)

    def mean(self):
        pi = self.pi.reshape(self.num_components, 1)
        return np.sum(self.mu * pi, 0)

    def condition(self, x, idx):
        """Builds a new GMM which is conditioned on an observation
        Based on the derivation in https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures
        
        The current code assumes you are conditioning on a contiguous
        region in the array starting from zero

        Arguments:
            x {np.array} -- observation to condition on
            idx {int} -- index array of the observation
        """
        mu_var = self.mu[:, idx:].copy()
        mu_obs = self.mu[:, :idx].copy()

        gmm_obs = GMM(mu_obs, self.pi, self.var_scale)
        gmm_var = GMM(mu_var, self.pi, self.var_scale)

        reweight = np.array([g.pdf(x) for g in gmm_obs.gaussians])
        reweight = gmm_var.pi * reweight
        reweight /= np.sum(reweight)
        gmm_var.pi = reweight

        for i in range(self.num_components):
            var_corr = self.var[i][idx:, :idx].dot(np.linalg.inv(self.var[i][:idx, :idx]))
            gmm_var.mu[i] += var_corr.dot((x - gmm_obs.mu[i]))
            gmm_var.var[i] -= var_corr.dot(self.var[i][:idx, idx:])
            try:
                gmm_var.init_pdf()
            except (np.linalg.linalg.LinAlgError, ValueError) as e:
                # find good recovery strategy, currently dificult
                # not perfect, fixes the eigenvalues
                gmm_var.var[i] += np.eye(gmm_var.dim) * (-1 * np.min(np.linalg.eig(gmm_var.var[i])[0]) + 10)
                gmm_var.init_pdf()
        return gmm_var

    def conditional_mean(self, x, idx):
        return self.condition(x, idx).mean()

    def conditional_sample(self, x, idx, n):
        return self.condition(x, idx).sample(n)

    def conditional_max(self, x, idx):
        cond = self.condition(x, idx)
        nn = np.argmax(cond.pi)
        return cond.mu[nn]

    def update_parameters(self, samples):
        converged = False

        while not converged:
            membership = self.e_step(samples)
            converged = self.m_step(samples, membership)

    def e_step(self, samples):
        prob = []
        for g, p in zip(self.gaussians, self.pi):
            prob.append(p * g.pdf(samples))
        prob = np.array(prob)
        prob = prob/np.sum(prob, 0)
        prob[np.isnan(prob)] = 1./self.num_components
        return prob.T

    def m_step(self, samples, membership):
        last_mu = self.mu.copy()
        last_var = self.var.copy()
        last_pi = self.pi.copy()

        new_pi = np.mean(membership, 0)
        norm = np.sum(membership, 0).reshape(-1, 1, 1)
        new_mu = np.array(
            [np.sum(membership[:, i:i+1] * samples, 0)/np.sum(membership[:, i])
                for i in range(self.num_components)])
        var_estimates = np.array([[
            (samples[j:j+1] - new_mu[i:i+1]).T.dot(samples[j:j+1] - new_mu[i:i+1])
            for i in range(self.num_components)] for j in range(samples.shape[0])])
        new_var = np.sum(membership.reshape(-1, self.num_components, 1, 1) * var_estimates, 0)/norm
        self.mu = new_mu
        self.pi = new_pi
        self.var = new_var
        try:
            self.init_pdf()
        except np.linalg.linalg.LinAlgError as e:
            new_var += np.eye(self.dim) * 1.e-5
            self.var = new_var
            self.init_pdf()
        converged = np.all(np.isclose(last_mu, new_mu))
        converged = converged and np.all(np.isclose(last_pi, new_pi))
        converged = converged and np.all(np.isclose(last_var, new_var))
        return converged

    def copy(self):
        return GMM(self.mu.copy(), self.pi.copy(), self.var_scale)


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

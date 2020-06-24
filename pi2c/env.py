"""
Environments to control
"""
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import random
import scipy as sc
import os

import torch
from torch.distributions import MultivariateNormal

import pi2c.env_def as env_def
import pi2c.env_autograd as env_autograd
from pi2c import jax_gmm


class BaseSim(object):

    env = None

    def run(self, policy):
        """Run the environment with a given policy and gather data.

        :policy
        :return
        """
        xt = np.zeros((self.duration, self.dim_xt))
        yt = np.zeros((self.duration, self.dim_yt))
        zt = np.zeros((self.duration+1, self.dim_y))
        x = self.init_env()
        for t in range(self.duration):
            u = policy(t, x)
            u = policy(t, x).reshape(self.dim_u,)
            x_prev = x
            x = self.forward(u)
            xt_entry = np.hstack((x_prev, u))
            xt[t, :] = xt_entry.squeeze()
            yt[t, :] = (x - x_prev).squeeze()
            zt[t, :]  = self.observe(x_prev.reshape((-1, 1)),
                u.reshape((-1, 1)))[0].squeeze()
        # terminal observation for terminal cost
        # assumes that zero is goal state
        zt[t+1, :] = self.observe(x.reshape((-1, 1)),
            np.zeros((self.dim_u, 1)))[0].squeeze()
        return xt, yt, zt

    def likelihood(self, x1, x0):
        '''
        Necessary for the particle filter smoothing algorithm
        '''
        raise NotImplementedError

    def sample(self, x, u):
        """
        SUPER HACKY, FIX with Joe!
        """
        x_cur = self.x
        self.x = x
        sample = self.forward(u)
        self.x = x_cur
        return sample

    def forward(self, u):
        raise NotImplementedError

    def observe(self, x, u):
        raise NotImplementedError

    def init_env(self):
        raise NotImplementedError

    def plot_sim(self, x, x_est=None, name="", res_dir=None):
        f, a = plt.subplots(self.dim_xt)
        a[0].set_title("Simulation: {}".format(name))
        a[-1].set_xlabel("Timesteps")
        for i, _a in enumerate(a):
            _a.plot(x[:, i], '.-', label="True")
            if x_est is not None:
                _a.plot(x_est[:, i], '.-', label="Planned")
            _a.legend()
            _a.set_ylabel("{} ({})".format(self.key[i], self.unit[i])
                          if self.unit[i] else "{}".format(self.key[i]))
        if res_dir:
            plt.savefig(os.path.join(res_dir, "sim_{}.png".format(name.replace(" ", "_"))),
                        bbox_inches='tight', format='png')
            plt.close(f)

    def close(self):
        if self.env is not None:
            self.env.close()
        else:
            print("Known environment, nothing to close")

class LinearSim(env_def.LinearDef, BaseSim):

    def __init__(self, duration):
        self.duration = duration
        self.a = self.a.reshape((-1)) # give me strength... # sorry Joe

    def init_env(self):
        self.x = np.copy(self.x0).squeeze()
        return self.x

    def forward(self, u):
        x = self.A.dot(self.x) + self.B.dot(u) + self.a
        self.x = x.reshape(self.x.shape)
        return self.x

class TorchLinearDisturbed(env_def.LinearDef, BaseSim):

    def __init__(self, duration):
        self.duration = duration
        self.a = torch.Tensor(self.a.reshape((-1, 1))) # give me strength...
        self.sig_x_noise = 0.001
        self.noise_pdf = sc.stats.multivariate_normal(np.zeros(2), self.sig_x_noise)
        self.A = torch.Tensor(self.A)
        self.B = torch.Tensor(self.B)
        self.normal = MultivariateNormal(torch.zeros(self.dim_x), torch.eye(self.dim_x) * self.sig_x_noise)

    def init_env(self, init_state_var, randomized=False):
        self.x = torch.tensor(np.copy(self.x0)).float()
        if randomized:
            init_state_noise = MultivariateNormal(torch.zeros(self.dim_x), torch.eye(self.dim_x) * init_state_var)
            self.x += init_state_noise.sample().reshape(self.x.shape)
        return self.x

    def transform_for_plot(self, x):
        return x

    def forward(self, u):
        x = self.A @ self.x + self.B @ u + self.a
        x += self.normal.sample((x.shape[1],)).squeeze().view(x.shape)
        return x

    def log_likelihood(self, x0, u, x1):
        _x = self.A @ x0 + self.B @ u + self.a
        mu = _x - x1
        return self.normal.log_prob(mu.T)


class LinearDisturbed(env_def.LinearDef, BaseSim):

    def __init__(self, duration):
        self.duration = duration
        self.a = self.a.reshape((-1, 1)) # give me strength...
        self.sig_x_noise = 0.001
        self.noise_pdf = sc.stats.multivariate_normal(np.zeros(2), self.sig_x_noise)
        self.key = random.PRNGKey(0)

    def init_env(self, init_state_var=1., randomized=False):
        self.x = self.x0.copy()
        if randomized:
            self.key, sk = random.split(self.key)
            self.x += random.normal(sk, self.x.shape) * init_state_var
        return self.x

    def transform_for_plot(self, x):
        return x

    def forward(self, u):
        self.key, sk = random.split(self.key)
        x = self.A @ self.x + self.B @ u + self.a
        r = random.normal(sk, x.shape) * self.sig_x_noise
        return x + r

    def log_likelihood(self, x0, u, x1):
        _x = self.A @ x0 + self.B @ u + self.a
        ll = jax_gmm.vec_log_normal_pdf(_x.T, np.eye(self.dim_x) * self.sig_x_noise, x1.T)
        return ll


class PendulumKnown(env_def.PendulumKnown, BaseSim):

    def __init__(self, duration):
        self.duration = duration
        self.key = random.PRNGKey(0)
        self.dim_x = 3

    def to_polar(self, x):
        x0 = np.arctan2(x[0, :], x[1, :])
        x1 = x[2, :]
        return np.vstack([x0, x1])

    def to_euclidean(self, x):
        x0 = np.sin(x[0, :])
        x1 = np.cos(x[0, :])
        x2 = x[1, :]
        return np.vstack([x0, x1, x2])

    def init_env(self, init_state_var=1., randomized=True):
        self.key, sk = random.split(self.key)
        self.x = self.x0
        if randomized:
            self.x += self.sigV.dot(random.normal(sk, self.x.shape))
        return self.to_euclidean(self.x)

    def forward(self, u):
        _x = self.to_polar(self.x)
        self.key, sk = random.split(self.key)
        disturbance = self.sigV.dot(random.normal(sk, _x.shape))
        self.x = self.to_euclidean(self.dynamics(_x, u) + disturbance)
        return self.x

    def transform_for_plot(self, x):
        return self.to_polar(x.T).T

    def log_likelihood(self, x0, u, x1):
        _x = self.dynamics(self.to_polar(x0), u)
        ll = jax_gmm.vec_log_normal_pdf(_x.T, self.sigV.dot(np.eye(2)), self.to_polar(x1).T)
        return ll


class TorchPendulumKnown(env_def.PendulumKnown, BaseSim):

    def __init__(self, duration):
        self.duration = duration
        self.key = random.PRNGKey(0)
        self.sigV = torch.tensor(self.sigV).float()
        self.normal = MultivariateNormal(torch.zeros(self.dim_x), self.sigV)

    def to_polar(self, x):
        x0 = torch.arctan2(x[:, 0]/x[:, 1])
        x1 = x[:, 2]
        return np.stack([x0, x1], 1)

    def to_euclidean(self, x):
        x0 = torch.sin(x[:, 0])
        x1 = torch.cos(x[:, 0])
        x2 = x[:, 1]
        return torch.stack([x0, x1, x2], 1)

    def init_env(self, init_state_var=1., randomized=False):
        self.x = torch.tensor(np.copy(self.x0)).float()
        if randomized:
            dist = MultivariateNormal(self.x.T, torch.eye(self.dim_x) * init_state_var)
            return dist.sample().T
        return self.to_euclidean(self.x)

    def forward(self, u):
        _x = self.to_polar(self.x)
        disturbance = self.normal.sample((_x.shape[1],)).T
        self.x = env_autograd.pendulum_dynamics_torch(_x, u) + disturbance
        return self.to_euclidean(self.x)

    def transform_for_plot(self, x):
        return self.to_polar(x.T).T

    def log_likelihood(self, x0, u, x1):
        _x = env_autograd.pendulum_dynamics_torch(self.to_polar(x0), u) - self.to_polar(x1)
        return self.normal.log_prob(_x.T)


class PendulumLinearObservationKnown(env_def.PendulumLinearObservationKnown, BaseSim):

    def __init__(self, duration):
        self.duration = duration

    def init_env(self):
        self.x = self.x0.squeeze()
        return self.x

    def forward(self, u):
        disturbance = self.sigV.dot(np.random.randn(self.dim_x,))
        self.x = self.dynamics(self.x, u) + disturbance
        return self.x

class PendulumSim(env_def.PendulumDef, BaseSim):

    def __init__(self, duration):
        self.duration = duration

    def init_env(self):
        self.env.reset()
        self.env.env.state = self.x0.squeeze()
        return self.env.env.state

    def forward(self, u):
        self.prev_th = self.env.env.state[0]
        x, r, fin, data = self.env.step(u)
        th = np.arctan2(x[1], x[0])
        if self.prev_th:
            _th = np.unwrap(np.array([self.prev_th, th]))
            th = _th[1]
        self.prev_th = th
        return np.array([th, x[2]])

class CartpoleKnown(env_def.CartpoleKnown, BaseSim):

    def __init__(self, duration):
        self.init_env()
        self.duration = duration

    def init_env(self):
        self.x = self.x0.squeeze()
        return self.x

    def forward(self, u):
        self.x = self.dynamics(self.x, u) + self.sigV.dot(np.random.randn(self.dim_x,))
        return self.x

class QuanserCartpoleKnown(env_def.QuanserCartpole, BaseSim):

    def __init__(self, duration):
        self.init_env()
        self.duration = duration

    def init_env(self):
        self.x = self.x0.squeeze()
        return self.x

    def forward(self, u):
        self.x = self.dynamics(self.x, u) + self.sigV.dot(np.random.randn(self.dim_x,))
        return self.x

class DoubleCartpoleKnown(env_def.DoubleCartpoleKnown, BaseSim):

    def __init__(self, duration):
        self.init_env()
        self.duration = duration

    def init_env(self):
        self.x = self.x0.squeeze()
        return self.x

    def forward(self, u):
        self.x = self.dynamics(self.x, u) + self.sigV.dot(np.random.randn(self.dim_x,))
        return self.x

class TwoLinkElasticJointRobotKnown(env_def.TwoLinkElasticJointRobotKnown, BaseSim):

    def __init__(self, duration):
        self.init_env()
        self.duration = duration

    def init_env(self):
        self.x = self.x0.squeeze()
        return self.x

    def forward(self, u):
        self.x = self.dynamics(self.x, u.reshape((-1, 1))).squeeze()
        return self.x

def make_env(exp):
    _lookup = {
        "LinearKnown": LinearSim,
        "LinearDisturbed": LinearDisturbed,
        "TorchLinearDisturbed": TorchLinearDisturbed,
        "PendulumKnown": PendulumKnown,
        "TorchPendulumKnown": TorchPendulumKnown,
        "PendulumLinearObservationKnown": PendulumLinearObservationKnown,
        "CartpoleKnown": CartpoleKnown,
        "QuanserCartpoleKnown": QuanserCartpoleKnown,
        "DoubleCartpoleKnown": DoubleCartpoleKnown,
        "ElasticTwoLinkKnown": TwoLinkElasticJointRobotKnown,
    }
    return _lookup[exp.ENVIRONMENT](exp.N_DURATION)

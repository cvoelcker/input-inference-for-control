"""
Environments to control
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import os

import pi2c.env_def as env_def


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
        x_cur = self.x.copy()
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
        self.a = self.a.reshape((-1,1)) # give me strength... # sorry Joe

    def init_env(self):
        self.x = np.copy(self.x0).squeeze()
        return self.x

    def forward(self, u):
        x = self.A.dot(self.x) + self.B.dot(u) + self.a
        self.x = x.reshape(self.x.shape)
        return self.x

class LinearDisturbed(env_def.LinearDef, BaseSim):

    def __init__(self, duration):
        self.duration = duration
        self.a = self.a.reshape((-1, 1)) # give me strength...
        self.sig_x_noise = 0.1
        self.noise_pdf = sc.stats.multivariate_normal(np.zeros(2), self.sig_x_noise)

    def init_env(self):
        self.x = np.copy(self.x0).squeeze()
        return self.x

    def forward(self, u):
        x = self.A.dot(self.x) + self.B.dot(u) + self.a + np.random.randn(*self.x.shape) * self.sig_x_noise
        self.x = x.reshape(self.x.shape)
        return self.x

    def likelihood(self, x0, u, x1):
        _x = self.A.dot(x0) + self.B.dot(u) + self.a
        _x -= x1
        return self.noise_pdf.pdf(_x.T)

class PendulumKnown(env_def.PendulumKnown, BaseSim):

    def __init__(self, duration):
        self.duration = duration

    def init_env(self):
        self.x = self.x0.squeeze()
        return self.x

    def forward(self, u):
        disturbance = self.sigV.dot(np.random.randn(self.dim_x,))
        self.x = self.dynamics(self.x, u) + disturbance
        return self.x

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
        "PendulumKnown": PendulumKnown,
        "PendulumLinearObservationKnown": PendulumLinearObservationKnown,
        "CartpoleKnown": CartpoleKnown,
        "QuanserCartpoleKnown": QuanserCartpoleKnown,
        "DoubleCartpoleKnown": DoubleCartpoleKnown,
        "ElasticTwoLinkKnown": TwoLinkElasticJointRobotKnown,
    }
    return _lookup[exp.ENVIRONMENT](exp.N_DURATION)

"""
Big list of TODO:
    - dependence on torch
        - alpha is disentangled
        - particles are sometimes explicitely casted to torch, remove
    - unify start state sampling and maybe work into env
        - small env wrapper for pytorch for consistency?
    - keep run weights and particles separate without going over RAM
    - implement different networks properly in poliy_torch
    - reimplementd KDE from git history
        - move policy update functions still in ParticleCell to one location and abstract
    - merge MoG policy/joint
    - implement alpha strategies
    - rework plotting
"""

import matplotlib.pyplot as plt
# import jax.numpy as np
import jax.scipy.special
import jax.numpy as np
import jax
from jax import random
import numpy as onp
import scipy.special
from copy import deepcopy
from contextlib import contextmanager
import pdb
import os
#import tikzplotlib
import dill
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch
from torch import nn, optim

import cProfile

from pi2c.utils import converged_list, finite_horizon_lqr, GaussianPrior
from pi2c.jax_gmm import GMM
from pi2c.cost_function import Cost2Prob
from pi2c.policy_torch import get_policy
from pi2c.score_matching import score_matching

def nop(it, *a, **k):
    return it

# tqdm = nop

# abstract over numerical functions for different backends
# necessary to code forward/backward for both torch and numpy
# based architectures
BACKEND = 'torch'
def set_numerical_backend(backend):
    if backend in ['torch', 'jax', 'numpy']:
        BACKEND = backend
    else:
        raise NotImplementedError('Backend {} unkown'.format(backend))

logsumexp_dict = {'torch': lambda x: torch.logsumexp(x, 0),
        'jax': lambda x: jax.scipy.special.logsumexp(x, 0),
        'numpy': scipy.special.logsumexp}
def logsumexp(x):
    return logsumexp_dict[BACKEND](x)


class ParticleI2cCell(nn.Module):

    def __init__(self, i, num_p, num_u_samples, alpha_init):
        """Initializes a particel swarm cell at a specific time index
        
        Arguments:
            i {int} -- time index of the cell
            env {env} -- the stochastic environment
            cost {cost} -- the cost function aka the observation probability
            num_p {int} -- number of particles
            M {int} -- number of backward trajectories for smoothing
        """
        super().__init__()
        self.i = i
        self.back_particle = None
        self.num_p = num_p
        self.alpha = alpha_init

        self.particles = None
        self.log_weights = None
        self.new_particles = None

        self.back_particles = None
        self.back_weights = None

        self.u_samples = num_u_samples

        self.key = random.PRNGKey(0)

    def set_policy(self, strategy, smoothing, policy_config):
        if strategy == 'VSMC':
            self.policy = get_policy(
                    policy_config.nn_type, 
                    self.dim_x, 
                    self.dim_u, 
                    policy_config.init_policy_mean, 
                    policy_config)
        elif strategy == 'mixture':
            self.policy = GMM(
                self.dim_x + self.dim_u, 
                policy_config.components, 
                self.dim_x, 
                policy_config.u_clipping,
                policy_config.init_policy_variance,
                policy_config.exp_factor)
            self.exp_factor = policy_config.exp_factor
        elif strategy == 'KDE':
            # TODO: setup KDE policy/joint
            raise NotImplementedError('[WIP] Implement')
        
        if smoothing in ['greedy', 'smoothing']:
            self.smoothing = smoothing
        else:
            raise ValueError('Smoothing strategy unknown')
        self.strategy = strategy

    def set_env(self, env, cost):
        self.env = env
        self.dim_u = env.dim_u
        self.dim_x = env.dim_x
        self.cost = cost
        self.obs_lik = Cost2Prob(self.cost)

    @property
    def forward_done(self):
        return self.particles is not None

    @property
    def backward_done(self):
        return self.back_particles is not None

    def forward_pass(self, particles, iteration, failed=False, alpha=1.):
        new_u = []
        new_u, mu, sig = self.policy(particles, self.u_samples)

        if self.strategy == 'VSMC':
            particles = torch.repeat_interleave(particles, self.u_samples, 0)
            samples, log_weights = self.obs_lik.log_sample(particles, new_u, self.num_p, alpha)
            samples.detach()
        elif self.strategy == 'mixture':
            u_weights = np.log(self.exp_factor) + np.exp(((-self.exp_factor**2 + 1)/(2 * self.exp_factor**2)) * ((new_u - mu)/sig)).reshape(-1,1)
            particles = np.repeat(particles, self.u_samples, 0)
            samples, log_weights = self.obs_lik.log_sample_jax(particles, new_u, self.num_p, u_weights, alpha)
        self.samples = samples//self.u_samples
        self.log_weights = log_weights[samples].squeeze()
        if self.strategy == 'VSMC':
            self.particles = torch.cat([particles, new_u], 1)[samples]
        elif self.strategy == 'mixture':
            self.particles = np.concatenate((particles, new_u), 1)[samples]
        self.new_particles = self.env.sample(particles[samples].T, new_u[samples].T).T
        return self.new_particles, self.particles, failed

    def greedy_backward_reweighing(self, particles, samples, weights):
        if self.strategy == 'mixture':
            samples = np.take(self.samples, samples)
        elif self.strategy == 'VSMC':
            self.log_weights = self.log_weights[samples] + weights
        return self.particles[samples], samples, self.log_weights

    def backward_smoothing(self, samples, unused, weights):
        """ Smoothing by backwards reweighing after Doucet et al
        """
        if self.strategy == 'VSMC':
            smoothed_weights = []
            for p in samples:
                p = p.reshape(1,-1)
                forward_ll = self.env.log_likelihood(self.particles[:, :self.dim_x].T, self.particles[:, self.dim_x:].T, p.T).reshape(-1)
                forward_ll_norm = logsumexp(self.log_weights + forward_ll)
                forward_ll_w_normalized = forward_ll - forward_ll_norm
                smoothed_weights.append(forward_ll_w_normalized)
            smoothed_weights = torch.tensor(smoothed_weights)
        elif self.strategy == 'mixture':
            def smoothing_loop(p):
                p = p.reshape(1,-1)
                forward_ll = self.env.log_likelihood(self.particles[:, :self.dim_x].T, self.particles[:, self.dim_x:].T, p[:, :self.dim_x].T).reshape(-1)
                forward_ll_norm = logsumexp(self.log_weights + forward_ll)
                return forward_ll - forward_ll_norm
            lls = jax.vmap(smoothing_loop)(samples)
            lls += weights.reshape(-1,1)
            smoothed_weights = logsumexp(lls)
        self.log_weights += smoothed_weights
        return self.particles, self.samples, self.log_weights

    def backward_pass(self, particles, samples, weights):
        if self.smoothing == 'greedy':
            return self.greedy_backward_reweighing(particles, samples, weights)
        elif self.smoothing == 'smoothing':
            return self.backward_smoothing(particles, samples, weights)
        else:
            raise NotImplementedError('Unknown smoothing strategy')

    def get_policy(self, x):
        """
        TODO: abstract away
        Extracts a policy from the posterior particle GMM
        using conditional max
        
        Arguments:
            x {np.ndarray} -- current position of the envtem
        """
        return self.policy.conditional_mean(x, self.dim_x)

    def update_policy(self, particles, weights):
        """
        """
        # resampled_particles = np.concatenate(particles)
        norm_log_weights = np.concatenate(weights) - np.max(weights)
        norm_weights = jax.nn.softmax(np.concatenate(weights))
        # resampled_particles = []
        # weights = np.concatenate(weights)
        # for p, w in zip(particles, weights):
        #     self.key, sk = random.split(self.key)
        #     samples = random.gumbel(sk, (len(p), len(p)))
        #     choices = np.argmax(samples + w.reshape(-1,1), 0)
        #     resampled_particles.append(np.take(p, choices, 0))
        # resampled_particles = np.concatenate(resampled_particles, 0)
        # self.policy.update_parameters(resampled_particles, np.zeros_like(weights))
        self.policy.update_parameters(np.concatenate(particles), norm_log_weights)

    def current_backward_costs(self):
        c = self.cost(self.back_particles[:, :self.dim_x], self.back_particles[:, self.dim_x:])
        return c.mean(), c.var()

    def save_state(self, save_path):
        torch.save(self.policy, save_path.format(self.i))

    def load_state(self, save_path):
        self.policy = torch.load(save_path.format(self.i))


class ParticleI2cGraph():
    
    def __init__(self, T, num_p, num_u_samples, M, mu_x0, sig_x0, alpha_init):
        """[summary]
        
        Arguments:
            env {[type]} -- [description]
            cost {[type]} -- [description]
            T {[type]} -- [description]
            num_p {[type]} -- [description]
            M {[type]} -- [description]
        """
        self.log_id = 0

        self.env = None
        self.cost = None
        self.T = T
        self.num_p = num_p
        self.M = M
        self.num_u_samples = num_u_samples
        self.num_f_p = self.num_p * self.num_u_samples
        self.alpha = alpha_init

        self.mu_x0 = mu_x0
        self.sig_x0 = sig_x0
        self.x0_dist = GaussianPrior(mu_x0, sig_x0)
        
        self.cells = []
        for t in range(T):
            self.cells.append(
                ParticleI2cCell(
                    t, num_p, num_u_samples, alpha_init))

        self.UPDATE_ALPHA = True

        self.policy_ready = False
        self.env_ready = False
        self.optimizer_ready = False

        self.numerical_backend = 'torch' # options are jax, numpy, torch

    @property
    def is_ready(self):
        return self.policy_ready and self.env_ready and self.optimizer_ready

    def set_policy(self, strategy, smoothing, parameters=None):
        assert self.env_ready, 'Tried to set policy before env'
        self.smoothing_strategy = smoothing
        self.policy_type = strategy
        for c in self.cells:
            c.set_policy(strategy, smoothing, parameters)
        if self.policy_type == 'mixture':
            self.gmm_components = parameters.components
            self.cost.torch = False
        self.policy_ready = True

    def set_optimizer(self, strategy, batch_size, clip_alpha, norm=None, lr=None, optimizer=None):
        assert self.policy_ready, 'Tried to set optimization before policy'
        self.optimizer_type = strategy
        if strategy == 'gradient':
            assert self.policy_type == 'VSMC', 'Gradient training only available for NN style policy'
            self.lr = lr
            self.norm = norm
            self.optimizer = optim.Adam(
                    [{'params': c.policy.parameters()} for c in self.cells
                     ], lr=self.lr)
            self.alpha_update = 'heuristic'
        if strategy == 'em':
            self.alpha_update = 'quadratic'
            self.sigXi0 = np.linalg.inv(self.cost.QR.numpy())
        self.clip_alpha = clip_alpha
        self.batch_size = batch_size
        self.optimizer_ready = True

    def set_env(self, env, cost):
        self.env = env
        self.cost = cost
        for c in self.cells:
            c.set_env(env, cost)

        # borrowed from the quadratic cost assumption
        # invalid for other costs
        # self.sigXi0 = np.linalg.inv(self.cost.QR)
        self.env_ready = True

    @property
    def _alpha(self):
        if self.policy_type == 'VSMC':
            return torch.tensor([float(self.alpha)])
        elif self.policy_type == 'mixture':
            return float(self.alpha)

    def run(self, it, max_iter, log_dir='log/', run_bimodal_exp=False):
        _iter = 0
        losses = []
        # alpha is updated only in the last timestep to prevent superfluous changes before policy convergence
        for _iter in tqdm(range(max_iter)):
            update_alpha = (_iter == (max_iter-1))
            forward_particles, weights, backward_particles = self._expectation(_iter, run_bimodal_exp)
            alpha, loss, converged = self._maximization(weights, backward_particles, update_alpha=update_alpha)
            if update_alpha:
                if self.clip_alpha:
                    self.alpha = np.clip(alpha, self.alpha * 0.91, self.alpha * 1.1)
                else:
                    self.alpha = alpha
            if self.policy_type == 'VSMC':
                losses.append(loss.detach().numpy())
        self.log_id += 1
        print()
        print(alpha)
        print(self.alpha)
        return self.alpha, loss, forward_particles, weights, backward_particles

    def _expectation(self, iteration, run_bimodal_exp):
        """Runs the forward, backward smoothing algorithm to estimate the
        current optimal state action trajectory
        
        Arguments:
            alpha {float} -- the current alpha optimization parameter
        """
        # sample initial x from the systems inital distribution
        all_weights = []
        all_forward_particles = []
        all_particles = []
        # run per cell loop
        for i in tqdm(range(self.batch_size)):
            final_sample, forward_particles = self.simulate_forward(self._alpha, run_bimodal_exp)
            all_forward_particles.append(forward_particles)
            if self.policy_type == 'VSMC':
                final_weights = self._alpha * self.cost(final_sample, torch.zeros((len(final_sample), 1)))
            elif self.policy_type == 'mixture':
                final_weights = self._alpha * self.cost(final_sample, np.zeros((len(final_sample), 1)))
            weights_backward, particles_backward = self.simulate_backwards(final_sample, final_weights)
            all_particles.append(particles_backward)
            all_weights.append(weights_backward)
        return all_forward_particles, all_weights, all_particles

    def simulate_forward(self, alpha, run_bimodal_exp):
        all_particles = []
        if run_bimodal_exp:
            particles_pos = self.x0_dist.sample(self.num_p//2)
            particles_neg = -self.x0_dist.sample(self.num_p//2)
            particles = np.concatenate((particles_pos, particles_neg), 0)
        else:
            particles = self.x0_dist.sample(self.num_p)
        if self.policy_type == 'VSMC':
            particles = torch.Tensor(particles)
        failed = False
        iteration = 0
        for c in self.cells:
            particles, sampled, failed = c.forward_pass(particles, iteration, failed, alpha)
            all_particles.append(c.particles)
        return particles, all_particles

    def simulate_backwards(self, samples, weights):
        all_weights = []
        samples = onp.arange(len(samples))
        particles = self.cells[-1].new_particles
        all_particles = []
        for c in reversed(self.cells):
            particles, samples, weights = c.backward_pass(particles, samples, weights)
            all_particles.append(particles)
            all_weights.append(c.log_weights)
        return all_weights, all_particles

    def _maximization(self, weights, particles, update_alpha=False):
        if self.policy_type == 'VSMC':
            particles = [torch.cat(p) for p in particles]
            logsumexp_weights = []
            for batch in weights:
                for w in batch:
                    logsumexp_weights.append(torch.logsumexp(w, 0))
            loss, converged = self._vsmc_maximization(logsumexp_weights)
            if update_alpha:
                np_particles = torch.cat(particles).detach().numpy()
                np_weights = np.concatenate([jax.nn.softmax(w.detach().numpy()) for w in weights[0]])
                alpha = self._score_matching_alpha_update(np_particles, np_weights)
                return alpha, loss, converged
            else:
                return None, loss, converged
        if self.policy_type == 'mixture':
            particles = np.array(particles)
            # weights = np.array(weights) - np.max(weights, -1, keepdims=True)
            weights = np.array(weights)
            for i, c in tqdm(list(enumerate(reversed(self.cells)))):
                c.update_policy(particles[:, i], weights[:, i])
            if update_alpha:
                np_particles = np.concatenate(np.concatenate(particles, -2), 0)
                np_weights = jax.nn.softmax(np.concatenate(weights, 1))
                alpha = self._score_matching_alpha_update(np_particles, np_weights)
                converged = False
                # alpha, converged = self._quadratic_alpha_update(self.alpha)
            else:
                converged = False
                alpha = self.alpha
            return alpha, None, converged

    def _vsmc_maximization(self, weights):
        ## JOINT UPDATE
        self.optimizer.zero_grad()
        loss = torch.stack(weights)
        loss = -torch.sum(loss)
        loss.backward()
        for c in self.cells:
            nn.utils.clip_grad_norm_(c.policy.parameters(), 100.)
        self.optimizer.step()
        converged = False
        return loss, converged

    def _heuristic_alpha_update(self, proposals, rounds):
        with torch.no_grad():
            alpha_proposals = torch.distributions.Normal(self.alpha, 0.5).sample((proposals, )).flatten()
            proposal_weights = []
            for i, alpha in enumerate(tqdm(alpha_proposals)):
                proposal_weights.append([])
                for j in range(rounds):
                    weights = self._expectation(None)
                    weights = torch.stack(weights).flatten()
                    weights = torch.mean(weights, 0)
                    proposal_weights[i].append(weights)
        proposal_weights = [torch.mean(torch.tensor(proposal_weights[i])) for i in range(proposals)]
        i = np.argmax(proposal_weights)
        return alpha_proposals[i]

    def _quadratic_alpha_update(self, alpha):
        # aka update alpha
        # this assumes a linear gaussian, or at least MoG joint
        # also has a time-varying alpha idea that was bad (but exciting)
        # we should probably recheck that with the particle filter, since it might
        # help with variance issues
        nan_traj = False
        s_covar = np.zeros_like(self.sigXi0)
        for i, c in enumerate(self.cells):
            for comp in range(self.gmm_components):
                if np.any(np.isnan(c.policy._mu[comp])):
                    print("y_m is nan")
                    nan_traj = True
                else:
                    err = c.policy._mu[comp] - self.cost.zg.numpy()
                    s_covar_t = (err.dot(err.T) + c.policy._sig[comp])
                    s_covar += c.policy._pi[comp] * s_covar_t

        s_covar = s_covar / float(self.T)
        s_covar = (s_covar + s_covar.T) / 2.0

        alpha_update = 1/(np.trace(np.linalg.solve(self.sigXi0, s_covar)) / float(self.env.dim_y))

        # error logging
        if nan_traj:
            print("ERROR, trajectory is nan")
        elif np.isnan(alpha_update):
            print("ERROR, alpha is nan, s_covar {}".format(s_covar))
        
        # actual alpha update
        else:
            if not alpha_update > 0.0:
                print("S covar bad")
                print(s_covar)
                print(np.linalg.det(s_covar))
                print(np.linalg.eig(s_covar))
                print(self.sigXi0)
                raise ValueError("{} <= 0.0".format(alpha_update))

            return alpha_update, np.isclose(alpha, alpha_update)
    
    def _score_matching_alpha_update(self, x, weights):
        alpha = score_matching(self.cost.cost_jax, x, weights.reshape(-1,1))
        return alpha
    
    def check_alpha_converged(self):
        return False

    def get_policy(self, x, i):
        return self.cells[i].policy.conditional_mean(x, 1)

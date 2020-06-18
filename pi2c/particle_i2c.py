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
import numpy as np
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


# abstract over numerical functions for different backends
# necessary to code forward/backward for both torch and numpy
# based architectures
BACKEND = 'torch'
def set_numerical_backend(backend):
    if backend in ['torch', 'jax', 'numpy']:
        BACKEND = backend
    else:
        raise NotImplementedError('Backend {} unkown'.format(backend))

logsumexp_dict = {'torch': torch.logsumexp,
        'jax': jax.scipy.special.logsumexp,
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

    def set_policy(self, strategy, smoothing, policy_config):
        if strategy == 'VSMC':
            self.policy = get_policy(
                    policy_config.nn_type, 
                    self.dim_x, 
                    self.dim_u, 
                    policy_config.init_policy_mean, 
                    policy_config)
        elif strategy == 'mixture':
            # TODO: setup GoM policy/joint
            raise NotImplementedError('[WIP] Implement')
        elif strategy == 'KDE':
            # TODO: setup KDE policy/joint
            raise NotImplementedError('[WIP] Implement')
        
        if smoothing in ['greedy', 'smoothing']:
            self.strategy = smoothing
        else:
            raise ValueError('Smoothing strategy unknown')

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
        new_u = self.policy(particles, self.u_samples)
        particles = torch.repeat_interleave(particles, self.u_samples, 0)
        samples, log_weights = self.obs_lik.log_sample(particles, new_u, self.num_p, alpha)
        samples = samples.detach()
        self.samples = samples//self.u_samples
        self.log_weights = log_weights[samples].squeeze()
        self.particles = torch.cat([particles, new_u], 1)[samples]
        self.new_particles = self.env.sample(particles[samples].T, new_u[samples].T).T
        assert not torch.any(torch.isnan(self.new_particles)), particles
        return self.new_particles, self.particles, failed

    def greedy_backward_reweighing(self, particles, samples, weights):
        samples = self.samples[samples]
        self.log_weights = self.log_weights[samples] + weights
        return self.particles[samples], samples, self.log_weights

    def backward_smoothing(self, samples, unused, weights):
        """ Smoothing by backwards reweighing after Doucet et al
        """
        smoothed_weights = []
        for p in self.particles:
            p = p.reshape(1,-1)
            forward_ll = self.env.log_likelihood(p[:, :self.dim_x].T, p[:, self.dim_x:].T, samples[:, :self.dim_x].T)
            forward_ll_w = forward_ll + weights
            forward_ll_norm = logsumexp(forward_ll + self.log_weights, 0)
            forward_ll_w_normalized = logsumexp(forward_ll_w - forward_ll_norm, 0)
            smoothed_weights.append(forward_ll_w_normalized)
        return self.particles, self.samples, torch.tensor(smoothed_weights)

    def backward_pass(self, particles, samples, weights):
        if self.strategy == 'greedy':
            return self.greedy_backward_reweighing(particles, samples, weights)
        elif self.strategy == 'smoothing':
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

    def update_policy(self):
        """
        TODO: abstract away
        Updates the prior for u by smoothing the posterior particle distribution 
        with a kernel density estimator
        """
        smoothed_posterior = GMM(self.back_particles, 
            np.ones(len(self.back_particles))/len(self.back_particles), 
            self.smooth_prior)
        self.policy = smoothed_posterior.marginalize(np.arange(self.dim_x, self.dim_x+self.dim_u))

    def current_backward_costs(self):
        c = self.cost(self.back_particles[:, :self.dim_x], self.back_particles[:, self.dim_x:])
        return c.mean(), c.var()

    def save_state(self, save_path):
        torch.save(self.policy, save_path.format(self.i))

    def load_state(self, save_path):
        self.policy = torch.load(save_path.format(i))


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
        self.policy_ready = True

    def set_optimizer(self, strategy, batch_size, norm=None, lr=None, optimizer=None):
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
        return torch.tensor([float(self.alpha)])

    def run(self, it, max_iter, log_dir='log/'):
        _iter = 0
        losses = []
        for _iter in tqdm(range(max_iter)):
            weights, particles = self._expectation(_iter)
            update_alpha = _iter == (max_iter-1)
            alpha, loss, converged = self._maximization(weights, particles, update_alpha=update_alpha)
            if update_alpha:
                #TODO: Find out why the sign error exists
                print()
                print()
                print(self.alpha)
                self.alpha = -alpha
            losses.append(loss.detach().numpy())
        np.savetxt('{}/losses_{}.npy'.format(log_dir, self.log_id), losses)
        for c in self.cells:
            c.save_state('{}/model_state_{}_'.format(log_dir, self.log_id) + '{}.torch')
        self.log_id += 1
        return self.alpha

    def _expectation(self, iteration):
        """Runs the forward, backward smoothing algorithm to estimate the
        current optimal state action trajectory
        
        Arguments:
            alpha {float} -- the current alpha optimization parameter
        """
        # sample initial x from the systems inital distribution
        all_weights = []
        all_particles = []
        # run per cell loop
        for i in range(self.batch_size):
            samples, all_samples = self.simulate_forward(self._alpha)
            weights = self.cells[-1].log_weights
            weights_backward = self.simulate_backwards(samples, weights)
            all_particles.append(all_samples)
            all_weights.append(weights_backward)
        return all_weights, all_particles

    def simulate_forward(self, alpha):
        all_particles = []
        particles = self.x0_dist.sample(self.num_p)
        particles = torch.Tensor(particles)
        failed = False
        iteration = 0
        for c in self.cells:
            particles, sampled, failed = c.forward_pass(particles, iteration, failed, torch.exp(alpha))
            all_particles.append(sampled)
        return particles, torch.cat(all_particles)

    def simulate_backwards(self, samples, weights):
        all_weights = []
        samples = np.arange(len(samples))
        particles = self.cells[-1].particles
        weights = self.cells[-1].log_weights
        for c in reversed(self.cells):
            particles, samples, weights = c.backward_pass(particles, samples, weights)
            all_weights.append(c.log_weights)
        return all_weights

    def _maximization(self, weights, particles, update_alpha=False):
        if self.policy_type == 'VSMC':
            logsumexp_weights = []
            for batch in weights:
                for w in batch:
                    logsumexp_weights.append(torch.logsumexp(w, 0))
            loss, converged = self._vsmc_maximization(logsumexp_weights)
            np_particles = torch.cat(particles).detach().numpy()
            if update_alpha:
                alpha = self._score_matching_alpha_update(np_particles)
            else:
                alpha = None
            return alpha, loss, converged
        
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
                    err = c.policy._mu[comp]  - self.cost.zg
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

            print('New alpha {}'.format(alpha_update))
            return alpha_update, np.isclose(alpha, alpha_update)
    
    def _score_matching_alpha_update(self, x):
        alpha = score_matching(self.cost.cost_jax, x)
        return alpha
    
    def check_alpha_converged(self):
        return False

    def get_policy(self, x, i):
        return self.cells[i].policy(x)

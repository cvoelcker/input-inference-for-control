import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.optimize import maximize
from copy import deepcopy
from contextlib import contextmanager
import pdb
import os
import matplotlib2tikz
import dill
from abc import ABC, abstractmethod

import cProfile

from pi2c.utils import converged_list, finite_horizon_lqr
from pi2c.cost_function import Cost2Prob


class ParticleI2cCell():

    def __init__(self, i, sys, cost, num_p, M, mu_u0, sig_u0, policy_smoothing):
        """Initializes a particel swarm cell at a specific time index
        
        Arguments:
            i {int} -- time index of the cell
            sys {env} -- the stochastic environment
            cost {cost} -- the cost function aka the observation probability
            num_p {int} -- number of particles
            M {int} -- number of backward trajectories for smoothing
        """
        self.i = i
        self.sys = sys
        self.cost = cost
        self.obs_lik = Cost2Prob(self.cost)
        self.back_particle
        self.num_p = num_p
        self.M = M

        self.particles = None
        self.weights = None
        self.new_particles = None

        self.back_particles = None
        self.back_weights = None

        self.u_prior = GaussianPrior(mu_u0, sig_u0)
        self.smoothing = policy_smoothing

    def forward_done(self):
        return self.particles is not None

    def backward_done(self):
        return self.back_particles is not None

    def forward(self, particles, alpha=1.):
        self.particles = self._sample(particles)
        self.weights = self._reweighing(alpha)
        self.new_particles = self._resample()
        return self.new_particles

    def _sample(self, particles):
        new_particles = self.sys.sample(particles)
        new_u = self.u_prior.sample(num_p)
        new_particles = np.concatenate([new_particles, new_u], 0)
        return new_particles

    def _reweighing(self, alpha):
        prob = self.obs_lik(self.particles, alpha)
        norm = np.sum(prob, 1)
        prob /= norm
        return prob

    def _resample(self):
        """Implements the resampling step of the forward particle filter
        
        Returns:
            np.ndarray -- array of resampled particles to propagate forward
        """
        # resample according to obs likelihood
        # This implements multinomial resampling for now
        samples = np.random.choice(
            np.arange(self.num_p), 
            size=self.num_p, 
            replace=True, 
            p=self.weights)
        new_particles = self.particles[:, samples]
        return new_particles

    def backward(self, particles):
        backwards = []
        smoothing_weights = []
        for p in particles:
            # get f(x_t+1|x_t)
            particle_likelihood = self.weights * self.sys.likelihood(p, self.particles)
            # renormalize
            particle_likelihood /= np.sum(particle_likelihood)
            # sample likely ancestor
            samples = np.random.choice(
                np.arange(self.num_p), 
                size=1, 
                replace=True, 
                p=particle_likelihood)
            # save backwards sample
            backwards.append(self.particles[samples])
            # save smoothing weights
            smoothing_weights.append(particle_likelihood)
        self.back_particles = np.array(backwards)
        self.back_weights = np.array(smoothing_weights)
        return np.array(self.back_particles)

    def policy(self, x):
        """[summary]
        
        Arguments:
            x {[type]} -- [description]
        """
        joint_prob = GMM(self.back_particles, 
            np.ones(len(self.back_particles))/len(self.back_particles), 
            self.smoothing)
        return joint_prob.conditional_mean(x, np.arange(self.sys.dim_sa))

    def update_u_prior(self):
        """Updates the prior for u by smoothing the posterior particle distribution 
        with a kernel density estimator
        """
        smoothed_posterior = GMM(self.particles, self.weights, self.smoothing)
        self.u_prior = smoothed_posterior.marginalize(np.arange(self.sys.dim_s, self.sys.dim_sa))


class ParticleI2cGraph():
    
    def __init__(self, sys, cost, T, num_p, M, mu_x0, sig_x0, policy_smoothing):
        """[summary]
        
        Arguments:
            sys {[type]} -- [description]
            cost {[type]} -- [description]
            T {[type]} -- [description]
            num_p {[type]} -- [description]
            M {[type]} -- [description]
        """
        self.sys = sys
        self.cost = cost
        self.T = T
        self.num_p = num_p
        self.M = M

        self.cells = []
        for t in range(T):
            self.cells.append(ParticleI2cCell(t, sys, cost, num_p, M, policy_smoothing))

        self.x0_dist = GaussianPrior(mu_x0, sig_x0)

        self.init_alpha = 1.

    def run(self):
        alpha = self.init_alpha
        self._expectation(alpha)
        alpha = self._maximization()

    def _expectation(self, alpha):
        # sample initial x from the systems inital distribution
        particles = self.x0_dist.sample(self.num_p)

        # run per cell loop
        for c in self.cells:
            particles = c.forward(particles)
        
        # initialize the backwards sample
        # here, the particles need to be weighted according to the final thing
        backwards_samples = np.random.choice(np.arange(self.M))
        particles = particles[backwards_samples]

        # run backwards smoothing via sampling
        for c in self.cells:
            particles = c.backward(particles)

    def _maximization(self, alpha):
        # get all cost parameters
        all_costs = []
        for c in self.cells:
            all_costs.append(self.cost(c.new_particles))
        all_costs = np.array(all_costs).flatten()

        # update alpha
        def log_lik(a):
            return a * np.sum(all_costs) + self.T * np.log(1./(np.sum(np.exp(a * all_costs))))
        alpha = maximize(log_lik, alpha, method='nelder-mead', 
            options={'xatol': 1e-8, 'disp': True})
        return alpha

    def get_policy(self, x, i):
        return self.cells[i].policy(x)
        


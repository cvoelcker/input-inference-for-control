import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from contextlib import contextmanager
import pdb
import os
#import tikzplotlib
import dill
from tqdm import tqdm
from abc import ABC, abstractmethod

import cProfile

from pi2c.utils import converged_list, finite_horizon_lqr, GaussianPrior, GMM
from pi2c.cost_function import Cost2Prob


class ParticleI2cCell():

    def __init__(self, i, sys, cost, num_p, M, mu_u0, sig_u0, smooth_prior, smooth_posterior):
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
        self.dim_u = sys.dim_u
        self.dim_x = sys.dim_x
        self.cost = cost
        self.obs_lik = Cost2Prob(self.cost)
        self.back_particle = None
        self.num_p = num_p
        self.M = M

        self.particles = None
        self.weights = None
        self.new_particles = None

        self.back_particles = None
        self.back_weights = None

        self.u_prior = GaussianPrior(mu_u0, sig_u0)
        self.smooth_prior = smooth_prior
        self.smooth_posterior = smooth_posterior

    def particles_x(self):
        return self.back_particles[:, :self.dim_x]

    def particles_u(self):
        return self.back_particles[:, self.dim_x:]

    def forward_done(self):
        return self.particles is not None

    def backward_done(self):
        return self.back_particles is not None

    def forward(self, particles, alpha=1.):
        new_u = self.u_prior.sample(self.num_p).reshape(-1, self.dim_u)
        # print(new_u)
        self.particles = np.concatenate([particles, new_u], 1)
        # print(particles)
        self.weights = self.obs_lik(particles, new_u, alpha)
        # print(self.weights)
        norm = np.sum(self.weights)
        # if norm == 0.:
        #     print('Found 0')
        self.weights /= norm
        samples = np.random.choice(
            np.arange(self.num_p), 
            size=self.num_p, 
            replace=True, 
            p=self.weights)
        new_particles = self.sys.sample(particles[samples].T, new_u[samples].T).T
        # print(new_particles)
        return new_particles

    def backward(self, particles):
        backwards = []
        smoothing_weights = []
        for p in particles:
            p = p.reshape(-1, 1)
            # get f(x_t+1|x_t)
            particle_likelihood = self.sys.likelihood(
                self.particles[:, :self.dim_x].T, self.particles[:, self.dim_x:].T, p)
            # renormalize
            particle_likelihood /= np.sum(particle_likelihood)
            # print(particle_likelihood)
            # sample likely ancestor
            samples = np.random.choice(
                np.arange(self.num_p), 
                size=1, 
                replace=True, 
                p=particle_likelihood)
            # save backwards sample
            backwards.append(self.particles[samples].reshape(-1))
            # save smoothing weights
            smoothing_weights.append(particle_likelihood)
        self.back_particles = np.array(backwards)
        self.back_weights = np.array(smoothing_weights)
        return np.array(self.back_particles)[:, :self.dim_x]

    def policy(self, x):
        """Extracts a policy from the posterior particle GMM
        using conditional max
        
        Arguments:
            x {np.ndarray} -- current position of the system
        """
        joint_prob = GMM(self.back_particles, 
            np.ones(len(self.back_particles))/len(self.back_particles), 
            self.smooth_posterior)
        return joint_prob.conditional_sample(x, np.arange(self.sys.dim_x))

    def update_u_prior(self):
        """Updates the prior for u by smoothing the posterior particle distribution 
        with a kernel density estimator
        """
        smoothed_posterior = GMM(self.back_particles, 
            np.ones(len(self.back_particles))/len(self.back_particles), 
            self.smooth_prior)
        self.u_prior = smoothed_posterior.marginalize(np.arange(self.dim_x, self.dim_x+self.dim_u))


class ParticleI2cGraph():
    
    def __init__(self, sys, cost, T, num_p, M, mu_x0, sig_x0, mu_u0, sig_u0, smooth_prior, smooth_posterior):
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
            self.cells.append(ParticleI2cCell(t, sys, cost, num_p, M, mu_u0, sig_u0, smooth_prior, smooth_posterior))

        self.x0_dist = GaussianPrior(mu_x0, sig_x0)

        self.init_alpha = 1.

    def run(self):
        alpha = self.init_alpha
        while True:
            print(alpha)
            self._expectation(alpha)
            next_alpha = self._maximization(alpha)
            if np.isclose(alpha, next_alpha):
                break
            alpha = next_alpha
        return alpha

    def _expectation(self, alpha):
        """Runs the forward, backward smoothing algorithm to estimate the
        current optimal state action trajectory
        
        Arguments:
            alpha {float} -- the current alpha optimization parameter
        """
        # sample initial x from the systems inital distribution
        particles = self.x0_dist.sample(self.num_p)

        # run per cell loop
        for c in tqdm(self.cells):
            particles = c.forward(particles, alpha)
        
        # initialize the backwards sample
        # here, the particles need to be weighted according to the final thing
        backwards_samples = np.random.choice(np.arange(self.num_p), self.M)
        particles = particles[backwards_samples]

        # run backwards smoothing via sampling
        for c in tqdm(list(reversed(self.cells))):
            particles = c.backward(particles)
            c.update_u_prior()
        
        all_costs = []
        for c in self.cells:
            all_costs.append(self.cost(c.particles_x(), c.particles_u()))
        all_costs = np.array(all_costs).flatten()
        print(np.mean(all_costs))

    def _maximization(self, alpha):
        # get all cost parameters
        all_costs = []
        for c in self.cells:
            all_costs.append(self.cost(c.particles_x(), c.particles_u()))
        all_costs = np.array(all_costs).flatten()

        # update alpha
        def log_lik(a):
            return -1 * (a * np.sum(all_costs) + self.T * np.log(1./(np.sum(np.exp(a * all_costs)))))

        alpha = minimize(log_lik, alpha, method='nelder-mead', 
            options={'xatol': 1e-8, 'disp': True}).x
        print(alpha)
        return alpha

    def get_policy(self, x, i):
        return self.cells[i].policy(x)

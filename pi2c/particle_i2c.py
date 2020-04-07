import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp
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

    def __init__(self, i, sys, cost, num_p, M, mu_u0, sig_u0=100., gmm_components=4):
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

        # build broad EM prior
        pi = np.array([1.] * gmm_components)/gmm_components
        # add small tither to GMM components to aid convergence of first round
        mu = np.tile(mu_u0.reshape(1, -1), (gmm_components, 1)) + \
             np.random.randn(gmm_components, len(mu_u0)) * 10.
        self.u_prior = GMM(mu, pi, sig_u0)

        self.alpha = 0.0001

    def particles_x(self):
        return self.back_particles[:, :self.dim_x]

    def particles_u(self):
        return self.back_particles[:, self.dim_x:]

    def forward_done(self):
        return self.particles is not None

    def backward_done(self):
        return self.back_particles is not None

    def forward(self, particles, alpha=1., use_time_alpha=False):
        new_u = []
        particles = np.repeat(particles, 10, 0)
        for i in range(self.num_p//10):
            u_prior = self.u_prior.condition(particles[i], self.dim_x)
            u = u_prior.sample(10).reshape(10, -1)
            new_u.append(u)
        # print(new_u)
        new_u = np.concatenate(new_u, 0)
        self.particles = np.concatenate([particles, new_u], 1)
        # print(particles)
        if use_time_alpha:
            self.weights = self.obs_lik(particles, new_u, self.alpha)
        else:
            self.weights = self.obs_lik(particles, new_u, alpha)
        norm = np.sum(self.weights)
        self.weights /= norm
        samples = np.random.choice(
            np.arange(self.num_p), 
            size=self.num_p//10, 
            replace=True, 
            p=self.weights)
        new_particles = self.sys.sample(particles[samples].T, new_u[samples].T).T
        # print(new_particles)
        return new_particles

    def backward(self, particles):
        backwards = []
        smoothing_weights = []
        all_samples = []
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
            all_samples.append(samples)
            # save backwards sample
            backwards.append(self.particles[samples].reshape(-1))
            # save smoothing weights
            smoothing_weights.append(particle_likelihood)
        print(len(np.unique(np.array(all_samples).reshape(-1))))
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
        return joint_prob.conditional_max(x, self.sys.dim_x)

    def update_u_prior(self):
        """Updates the prior for u by smoothing the posterior particle distribution 
        with a kernel density estimator
        """
        smoothed_posterior = GMM(self.back_particles, 
            np.ones(len(self.back_particles))/len(self.back_particles), 
            self.smooth_prior)
        self.u_prior = smoothed_posterior.marginalize(np.arange(self.dim_x, self.dim_x+self.dim_u))

    def update_alpha(self):
        all_costs = self.cost(self.particles_x(), self.particles_u())
        # self.alpha = np.sqrt(self.M/(costs**2))

        def alpha_eq(a):
            return np.sum(a * all_costs) - logsumexp(a * all_costs)

        def alpha_der(a):
            return np.sum(all_costs) - (np.sum(all_costs*np.exp(a*all_costs)))/(np.sum(np.exp(a*all_costs)))

        # print(alpha_eq(self.alpha))
        
        if alpha_eq(self.alpha*1.1) > alpha_eq(self.alpha):
            self.alpha *= 1.1
        elif alpha_eq(self.alpha/1.1) > alpha_eq(self.alpha):
            self.alpha /= 1.1

        # import matplotlib.pyplot as plt

        # x = np.linspace(0.00001, 0.0001, 1000)
        # y = [alpha_eq(_x) for _x in x]
        # print(x,y)
        # plt.plot(x,y)
        # plt.savefig('test.png')
        # quit()


class ParticleI2cGraph():
    
    def __init__(self, sys, cost, T, num_p, M, mu_x0, sig_x0, mu_u0, sig_u0, gmm_components=1):
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
            self.cells.append(ParticleI2cCell(t, sys, cost, num_p, M, mu_u0, sig_u0, gmm_components))

        self.x0_dist = GaussianPrior(mu_x0, sig_x0)

        self.init_costs()

    def init_costs(self):
        num_samples = 10000

        s_grid = np.random.uniform(-100, 100, (num_samples, self.sys.dim_x))
        a_grid = np.random.uniform(-100, 100, (num_samples, self.sys.dim_u))

        self.sample_costs = self.cost(s_grid, a_grid)

    def run(self, init_alpha, use_time_alpha=False):
        if use_time_alpha:
            for c in self.cells:
                c.alpha = init_alpha
        alpha = init_alpha
        # self._expectation(alpha, use_time_alpha)
        while True:
            # print(alpha)
            self._expectation(alpha, use_time_alpha)
            next_alpha = self._maximization(alpha, use_time_alpha)
            if use_time_alpha:
                print(np.mean([c.alpha for c in self.cells]))
                if self.check_alpha_converged():
                    break
            elif np.isclose(alpha, next_alpha):
                break
            alpha = next_alpha
        return alpha

    def _expectation(self, alpha, use_time_alpha=False):
        """Runs the forward, backward smoothing algorithm to estimate the
        current optimal state action trajectory
        
        Arguments:
            alpha {float} -- the current alpha optimization parameter
        """
        # sample initial x from the systems inital distribution
        particles = self.x0_dist.sample(self.num_p//10)

        # run per cell loop
        for c in tqdm(self.cells):
            particles = c.forward(particles, alpha, use_time_alpha)
        
        # initialize the backwards sample
        # here, the particles need to be weighted according to the final thing
        backwards_samples = np.random.choice(np.arange(self.num_p//10), self.M)
        particles = particles[backwards_samples]

        # run backwards smoothing via sampling
        for c in tqdm(list(reversed(self.cells))):
            particles = c.backward(particles)
            # c.update_u_prior()
        
        all_costs = []
        for c in self.cells:
            all_costs.append(self.cost(c.particles_x(), c.particles_u()))
        all_costs = np.array(all_costs).flatten()
        print(np.mean(all_costs))

    def _maximization(self, alpha, use_time_alpha=False):
        ## JOINT UPDATE
        for c in reversed(self.cells):
            # print(np.var(c.back_particles, 0))
            c.u_prior.update_parameters(c.back_particles)
        ## ALPHA UPDATE


        # # get all cost parameters
        # if use_time_alpha:
        #     for c in self.cells:
        #         c.update_alpha()
        #     return 1.
        # else:
        #     all_costs = []
        #     for c in self.cells:
        #         all_costs.append(np.sum(self.cost(c.particles_x(), c.particles_u())))
        #     all_costs = np.array(all_costs)

        #     def alpha_eq(a):
        #         return self.T * (np.mean(a * all_costs) - np.log(np.mean(np.exp(a * self.sample_costs))))

        #     def alpha_der(a):
        #         return self.T * (np.mean(all_costs) - (np.mean(self.sample_costs*np.exp(a*self.sample_costs)))/(np.mean(np.exp(a*self.sample_costs))))

        #     new_alpha = alpha + 1e-12 * alpha_der(alpha)

        #     print(alpha_der(alpha))
        #     print(alpha_eq(alpha))
        #     print(alpha_eq(0))
        #     print(alpha_eq(-0.001))

        #     print(new_alpha)

        #     assert (alpha_eq(new_alpha) >= alpha_eq(alpha)) and new_alpha > 0, '{} {} {}'.format(alpha_der(alpha), alpha_eq(new_alpha), alpha_eq(alpha))

        #     return new_alpha
        return alpha

    def check_alpha_converged(self):
        return False

    def get_policy(self, x, i):
        return self.cells[i].policy(x)

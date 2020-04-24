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

from pi2c.utils import converged_list, finite_horizon_lqr, GaussianPrior
from pi2c.jax_gmm import GMM
from pi2c.cost_function import Cost2Prob


class ParticleI2cCell():

    def __init__(self, i, sys, cost, num_p, M, mu_u0, sig_u0, alpha_init, gmm_components=4, u_samples=10):
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
        self.alpha = alpha_init

        self.particles = None
        self.weights = None
        self.new_particles = None

        self.back_particles = None
        self.back_weights = None

        # build broad EM prior
        self.xu_joint = GMM(self.dim_x + self.dim_u, gmm_components)

        self.u_samples = u_samples

    def particles_x(self):
        return self.back_particles[:, :self.dim_x]

    def particles_u(self):
        return self.back_particles[:, self.dim_x:]

    def forward_done(self):
        return self.particles is not None

    def backward_done(self):
        return self.back_particles is not None

    def forward(self, particles, iteration, failed=False, alpha=1., use_time_alpha=False):
        new_u = []
        new_u = self.xu_joint.conditional_sample(particles, self.dim_x, self.u_samples)
        particles = np.repeat(particles, self.u_samples, 0)
        # particles = np.repeat(particles, self.u_samples, 0)
        # for i in range(self.num_p//self.u_samples):
        #     xu_joint = self.xu_joint.condition(particles[i], self.dim_x)
        #     u = xu_joint.sample(self.u_samples).reshape(self.u_samples, -1)
        #     new_u.append(u)
        # new_u = np.concatenate(new_u, 0)
        if use_time_alpha:
            self.weights = self.obs_lik(particles, new_u, self.alpha)
        else:
            self.weights = self.obs_lik(particles, new_u, alpha)
        norm = np.sum(self.weights)
        # if all particle weights are numerically impossible to compute
        # the run is seen as failed
        if norm == 0.:
            failed = True
        self.weights /= norm
        samples = np.random.choice(
            np.arange(self.num_p), 
            size=self.num_p//self.u_samples, 
            replace=True, 
            p=self.weights)
        self.particles = np.concatenate([particles, new_u], 1)[samples]
        new_particles = self.sys.sample(particles[samples].T, new_u[samples].T).T
        return new_particles, failed

    def backward(self, particles):
        backwards = []
        smoothing_weights = []
        all_samples = []
        for p in particles:
            p = p[:self.dim_x].reshape(-1, 1)
            particle_likelihood = self.sys.likelihood(
                self.particles[:, :self.dim_x].T, self.particles[:, self.dim_x:].T, p)
            # renormalize
            particle_likelihood /= np.sum(particle_likelihood)
            # sample likely ancestor
            samples = np.random.choice(
                np.arange(self.num_p//self.u_samples), 
                size=1, 
                replace=True, 
                p=particle_likelihood)
            all_samples.append(samples)
            # save backwards sample
            backwards.append(self.particles[samples].reshape(-1))
            # save smoothing weights
            smoothing_weights.append(particle_likelihood)
        self.back_particles = np.array(backwards)
        self.back_weights = np.array(smoothing_weights)
        return np.array(self.back_particles)

    def policy(self, x):
        """Extracts a policy from the posterior particle GMM
        using conditional max
        
        Arguments:
            x {np.ndarray} -- current position of the system
        """
        return self.xu_joint.conditional_mean(x, self.dim_x)

    def update_xu_joint(self):
        """Updates the prior for u by smoothing the posterior particle distribution 
        with a kernel density estimator
        """
        smoothed_posterior = GMM(self.back_particles, 
            np.ones(len(self.back_particles))/len(self.back_particles), 
            self.smooth_prior)
        self.xu_joint = smoothed_posterior.marginalize(np.arange(self.dim_x, self.dim_x+self.dim_u))

    def update_alpha(self):
        """Not done
        """
        all_costs = self.cost(self.particles_x(), self.particles_u())

        def alpha_eq(a):
            return np.sum(a * all_costs) - logsumexp(a * all_costs)

        def alpha_der(a):
            return np.sum(all_costs) - (np.sum(all_costs*np.exp(a*all_costs)))/(np.sum(np.exp(a*all_costs)))
    
    def current_backward_costs(self):
        c = self.cost(self.back_particles[:, :self.dim_x], self.back_particles[:, self.dim_x:])
        return c.mean(), c.var()


class ParticleI2cGraph():
    
    def __init__(self, sys, cost, T, num_p, M, mu_x0, sig_x0, mu_u0, sig_u0, alpha_init, gmm_components=1, u_samples=100, num_runs=10):
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
        self.u_samples = u_samples
        self.alpha = alpha_init

        self.cells = []
        for t in range(T):
            self.cells.append(ParticleI2cCell(t, sys, cost, num_p, M, mu_u0, sig_u0, alpha_init, gmm_components, u_samples))

        self.gmm_components = gmm_components
        self.x0_dist = GaussianPrior(mu_x0, sig_x0)
        self.num_runs = num_runs
        
        # borrowed from the quadratic cost assumption
        self.sigXi0 = np.linalg.inv(self.cost.QR)

        self.UPDATE_ALPHA = True

    def init_costs(self):
        """Get an MC estimate of the current cost function integral (currently not used)
        """
        num_samples = 10000

        s_grid = np.random.uniform(-100, 100, (num_samples, self.sys.dim_x))
        a_grid = np.random.uniform(-100, 100, (num_samples, self.sys.dim_u))

        self.sample_costs = self.cost(s_grid, a_grid)

    def run(self, init_alpha, use_time_alpha=False, max_iter=1000):
        # if use_time_alpha:
        #     for c in self.cells:
        #         c.alpha = init_alpha
        alpha = init_alpha
        _iter = 0
        while True and _iter < max_iter:
            self._expectation(init_alpha, _iter, use_time_alpha)
            next_alpha, converged = self._maximization(init_alpha, use_time_alpha)
            alpha = np.clip(next_alpha, 0.5 * alpha, 2 * alpha)
            if use_time_alpha and self.check_alpha_converged():
                    break
            elif converged:
                break
            _iter += 1
        return alpha

    def _expectation(self, alpha, iteration, use_time_alpha=False):
        """Runs the forward, backward smoothing algorithm to estimate the
        current optimal state action trajectory
        
        Arguments:
            alpha {float} -- the current alpha optimization parameter
        """
        all_samples = []
        failed_counter = 0
        i = self.num_runs
        num_seq_failed = 0
        with tqdm(total=self.num_runs) as pbar:
            while i > 0:
                run_samples = []
                # sample initial x from the systems inital distribution
                particles = self.x0_dist.sample(self.num_p//self.u_samples)
                
                failed = False
                # run per cell loop
                for c in self.cells:
                    n_particles, failed = c.forward(particles, iteration, failed, alpha, use_time_alpha)
                    # check if all particle trajectories have a numeric probability of 0
                    if failed:
                        num_seq_failed += 1
                        break
                    particles = n_particles
                if not failed:
                    num_seq_failed = 0
                    # initialize the backwards sample
                    # here, the particles need to be weighted according to the final thing
                    # (actually, they don't really, since they where weighted in the final round of the forward
                    # pass)
                    if self.M > self.num_p//self.u_samples:
                        self.M = self.num_p//self.u_samples
                    backwards_samples = np.random.choice(np.arange(self.num_p//self.u_samples), self.M, replace=False)
                    particles = particles[backwards_samples]
                    
                    # run backwards smoothing via sampling
                    for c in list(reversed(self.cells)):
                        particles = c.backward(particles)
                        run_samples.append(particles)
                    all_samples.append(run_samples)
                    i -= 1
                    pbar.update(1)
        all_samples = np.concatenate(all_samples, 1)

        for i, c in enumerate(list(reversed(self.cells))):
            c.back_particles = all_samples[i]
        
    def _maximization(self, alpha, use_time_alpha=False):
        ## JOINT UPDATE
        ll = 0.
        v = 0.
        for i, c in tqdm(list(enumerate(self.cells))):
            c.xu_joint.update_parameters(c.back_particles)
            r = c.current_backward_costs()
            ll += r[0]
            v  += r[1]
        print(self.cells[-1].back_particles.mean(0))
        print(self.cells[-1].back_particles.var(0))
        print(ll/100.)
        print(v/100.)
        if self.UPDATE_ALPHA:
            alpha, converged = self._alpha_update(alpha, use_time_alpha)
        return alpha, converged

    def _alpha_update(self, alpha, use_time_alpha=False):
        # aka update alpha
        # also has a time-varying alpha idea that was bad (but exciting)
        # we should probably recheck that with the particle filter, since it might
        # help with variance issues
        nan_traj = False
        s_covar = np.zeros_like(self.sigXi0)
        for i, c in enumerate(self.cells):
            for comp in range(self.gmm_components):
                if np.any(np.isnan(c.xu_joint.mu[comp])):
                    print("y_m is nan")
                    nan_traj = True
                else:
                    err = c.xu_joint.mu[comp]  - self.cost.zg
                    s_covar_t = (err.dot(err.T) + c.xu_joint.sig[comp])
                    s_covar += c.xu_joint.pi[comp] * s_covar_t

            if use_time_alpha:
                s_covar = (s_covar + s_covar.T) / 2.0
                c_alpha = 1/(np.trace(np.linalg.solve(self.sigXi0, s_covar)) / float(self.sys.dim_y))
                c.alpha = np.clip(c_alpha, 0.66*c.alpha, 1.5*c.alpha)
                s_covar = np.zeros_like(s_covar)

                    # logging, figure out later
                    # tv = np.trace(np.linalg.solve(self.sigXi0, s_covar_t)) / float(self.sys.dim_y)
                    # self.alpha_tv[i] = np.clip(tv, 0.5, 50)
        s_covar = s_covar / float(self.T)
        s_covar = (s_covar + s_covar.T) / 2.0

        if use_time_alpha:
            alpha_update = np.mean([c.alpha for c in self.cells])
        else:
            alpha_update = 1/(np.trace(np.linalg.solve(self.sigXi0, s_covar)) / float(self.sys.dim_y))

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
                # self.plot_traj(
                #     -1, dir_name=self.res_dir,
                #     filename="bad_alpha")
                raise ValueError("{} <= 0.0".format(alpha_update))

            print('New alpha {}'.format(alpha_update))
            return alpha_update, np.isclose(alpha, alpha_update)


            # get into update ratio reasoning later, might be crucial for particle
            # variance reasons, especially in the early steps

            # alpha_update_ratio = alpha_update / self.alpha
            # self.alpha_update_tol_u = 2. - self.alpha_update_tol
            # if alpha_update_ratio < self.alpha_update_tol:
            #     alpha_update = self.alpha_update_tol  * self.alpha
            # if alpha_update_ratio > self.alpha_update_tol_u:
            #     alpha_update = self.alpha_update_tol_u * self.alpha

            # more logging and i22c specific updates
            # self.alphas.append(alpha_update)
            # self.alphas_tv.append(np.copy(self.alpha_tv))
            # self.alpha = alpha_update
            # lamXi = np.linalg.inv(self.sigXi)
            # for c in self.cells:
            #     c.sigXi = self.sigXi
            #     c.lamXi = lamXi

    def check_alpha_converged(self):
        return False

    def get_policy(self, x, i):
        return self.cells[i].policy(x)

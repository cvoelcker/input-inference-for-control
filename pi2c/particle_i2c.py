import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# import numpy as np
import jax.numpy as np
import numpy as onp
from jax import random, jit, vmap, grad, ops
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

    def __init__(self, i, sys, cost, num_p, M, alpha_init, gmm_components=4, u_samples=10, random_seed=1):
        """Initializes a particel swarm cell at a specific time index
        
        Arguments:
            i {int} -- time index of the cell
            sys {env} -- the stochastic environment
            cost {cost} -- the cost function aka the observation probability
            num_p {int} -- number of particles
            M {int} -- number of backward trajectories for smoothing
        """
        self.key = random.PRNGKey(random_seed)

        self.i = i
        self.sys = sys
        self.dim_u = sys.dim_u
        self.dim_x = sys.dim_x
        self.cost = cost
        self.obs_lik = Cost2Prob(self.cost, num_p, num_p//u_samples)
        self.back_particle = None
        self.num_p = num_p
        self.M = M
        self.alpha = alpha_init

        self.particles = None
        self.log_weights = None
        self.new_particles = None

        self.back_particles = None
        self.back_weights = None

        # build broad EM prior
        self.xu_joint = GMM(self.dim_x + self.dim_u, gmm_components)

        self.u_samples = u_samples

    def copy(self):
        new_cell = ParticleI2cCell(self.i, self.sys, self.cost, self.num_p, self.M, self.mu_u0, self.sig_u0, self.alpha, self.gmm_components, self.u_samples)
        new_cell.xu_joint = self.xu_joint.copy()
        new_cell.particles = self.particles.copy()
        new_cell.log_weights = self.log_weights.copy()
        new_cell.new_particles = self.new_particles.copy()
        new_cell.back_particles = self.back_particles.copy()
        new_cell.back_weights = self.back_weights.copy()
        return new_cell

    def particles_x(self):
        return self.back_particles[:, :self.dim_x]

    def particles_u(self):
        return self.back_particles[:, self.dim_x:]

    def forward_done(self):
        return self.particles is not None

    def backward_done(self):
        return self.back_particles is not None

    #@jit
    def forward(self, particles, iteration, failed=False, alpha=1., use_time_alpha=False):
        new_u = []
        # new_u = self.xu_joint.conditional_sample(particles, self.dim_x, self.u_samples)
        new_u = np.zeros((self.num_p, 1))
        assert not np.any(np.isnan(new_u)), new_u
        particles = np.repeat(particles, self.u_samples, 0)
        if use_time_alpha:
            samples, self.log_weights = self.obs_lik.log_sample(particles, new_u, self.alpha)
        else:
            samples, self.log_weights = self.obs_lik.log_sample(particles, new_u, alpha)
        self.forward_samples = samples
        self.particles = np.concatenate([particles, new_u], 1)
        assert not np.any(np.isnan(self.particles))
        self.new_particles = self.sys.simulate(np.take(particles,samples,0).T, np.take(new_u, samples,0).T).T
        return self.new_particles, self.particles, failed
    
    #@jit
    def backward(self, particles):
        backwards = []
        smoothing_weights = []
        all_samples = []
        dim_x = self.dim_x
        f_particles = self.particles[:, :dim_x].T
        f_u_particles = self.particles[:, dim_x:].T
        num_p = self.num_p
        def get_particle_likelihood(p):
            p = p[:self.dim_x].reshape(-1, 1)
            particle_log_likelihood = self.sys.log_likelihood(f_particles, f_u_particles, p)
            # particle_log_likelihood = np.log(particle_likelihood)
            # renormalize
            # particle_likelihood /= np.sum(particle_likelihood)
            # sample likely ancestor
            # sample = random.categorical(
            #         self.key,
            #         particle_log_likelihood,
            #         shape = (1,))
            max_sample = random.gumbel(self.key, (num_p,))
            sample = np.argmax(particle_log_likelihood.flatten() + max_sample)
            return particle_log_likelihood.flatten()[sample], sample
            
        vectorized = vmap(get_particle_likelihood)
        smoothing_weights, all_samples = vectorized(particles)
        
        backwards = self.particles[all_samples]
        # print(all_samples)
        # save smoothing weights

        self.backwards_samples = all_samples
        self.log_weights_back = np.take(self.log_weights, all_samples, 0)
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
    
    def __init__(self, sys, cost, T, num_p, M, mu_x0, sig_x0, mu_u0, sig_u0, alpha_init, gmm_components=1, u_samples=100, num_runs=10, random_seed=1):
        """[summary]
        
        Arguments:
            sys {[type]} -- [description]
            cost {[type]} -- [description]
            T {[type]} -- [description]
            num_p {[type]} -- [description]
            M {[type]} -- [description]
        """
        self.key = random.PRNGKey(random_seed)
        self.sys = sys
        self.cost = cost
        self.T = T
        self.num_p = num_p
        self.M = min(M, num_p//u_samples)
        self.u_samples = u_samples
        self.num_f_p = self.num_p//self.u_samples
        self.alpha = alpha_init

        self.mu_x0 = mu_x0
        self.mu_u0 = mu_u0
        self.sig_x0 = sig_x0
        self.sig_u0 = sig_u0

        self.cells = []
        for t in range(T):
            self.cells.append(ParticleI2cCell(t, sys, cost, num_p, M, alpha_init, gmm_components, u_samples, random_seed))

        self.gmm_components = gmm_components
        self.x0_dist = GaussianPrior(mu_x0, sig_x0)
        self.num_runs = num_runs
        
        # borrowed from the quadratic cost assumption
        # invalid for other costs
        self.sigXi0 = np.linalg.inv(self.cost.QR)

        self.UPDATE_ALPHA = True

    @property
    def is_multimodal(self):
        return self.gmm_components > 1

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
            next_alpha, converged = self._maximization(alpha, use_time_alpha)
            # alpha = np.clip(next_alpha, 0.5 * alpha, 2 * alpha)
            if use_time_alpha and self.check_alpha_converged():
                    break
            elif converged:
                break
            _iter += 1
        return next_alpha

    def _expectation(self, alpha, iteration, use_time_alpha=False):
        """Runs the forward, backward smoothing algorithm to estimate the
        current optimal state action trajectory
        
        Arguments:
            alpha {float} -- the current alpha optimization parameter
        """
        all_samples = []
        all_f_samples = []
        failed_counter = 0
        i = self.num_runs
        num_seq_failed = 0
        with tqdm(total=self.num_runs) as pbar:
            while i > 0:
                run_samples = []
                run_f_samples = []
                # sample initial x from the systems inital distribution
                particles = self.x0_dist.sample(self.num_p//self.u_samples)
                
                neg = random.randint(self.key, (len(particles)//2,), 0, len(particles))
                particles = ops.index_update(particles, neg, np.take(-particles, neg, 0))

                failed = False
                # run per cell loop
                for c in self.cells:
                    particles, sampled, failed = c.forward(particles, iteration, failed, alpha, use_time_alpha)
                    run_f_samples.append(sampled.copy())
                    # check if all particle trajectories have a numeric probability of 0
                    if failed:
                        num_seq_failed += 1
                        break
                if not failed:
                    num_seq_failed = 0
                    # initialize the backwards sample
                    # here, the particles need to be weighted according to the final thing
                    # (actually, they don't really, since they where weighted in the final round of the forward
                    # pass), we went one step "too far"
                    # due to no reweigh, I'm aslo sampling without replacing
                    if self.M > self.num_p//self.u_samples:
                        self.M = self.num_p//self.u_samples
                    backwards_samples = random.randint(self.key, (self.M,), 0, self.num_p//self.u_samples)
                    particles = np.take(particles, backwards_samples, axis=0)
                    
                    # run backwards smoothing via sampling
                    for c in list(reversed(self.cells)):
                        particles = c.backward(particles)
                        run_samples.append(particles.copy())
                    all_samples.append(np.array(run_samples))
                    all_f_samples.append(np.array(run_f_samples))
                    i -= 1
                    pbar.update(1)
        self.all_samples = np.concatenate(all_samples, 1)
        self.all_f_samples = np.concatenate(all_f_samples, 1)

        for i, c in enumerate(list(reversed(self.cells))):
            c.back_particles = self.all_samples[i]
            # print(c.back_particles.shape)
        
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
                if np.any(np.isnan(c.xu_joint._mu[comp])):
                    print("y_m is nan")
                    nan_traj = True
                else:
                    err = c.xu_joint._mu[comp]  - self.cost.zg
                    s_covar_t = (err.dot(err.T) + c.xu_joint._sig[comp])
                    s_covar += c.xu_joint._pi[comp] * s_covar_t

            if use_time_alpha:
                s_covar = (s_covar + s_covar.T) / 2.0
                c_alpha = 1/(np.trace(np.linalg.solve(self.sigXi0, s_covar)) / float(self.sys.dim_y))
                #c.alpha = np.clip(c_alpha, 0.66*c.alpha, 1.5*c.alpha)
                c.alpha = c_alpha
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
                raise ValueError("{} <= 0.0".format(alpha_update))

            print('New alpha {}'.format(alpha_update))
            return alpha_update, np.isclose(alpha, alpha_update)
    
    def check_alpha_converged(self):
        return False

    def get_policy(self, x, i):
        return self.cells[i].policy(x)

    def copy(self):
        new_graph = ParticleI2cGraph(self.sys, 
                self.cost, self.T, self.num_p, 
                self.M, self.mu_x0, self.sig_x0, 
                self.mu_u0, self.sig_u0, self.alpha_init, 
                self.gmm_components, self.u_samples, 
                self.num_runs)
        for i, c in enumerate(self.cells):
            new_graph.cells[i] = c.copy()
        return new_graph


class ParticlePlotter():

    def __init__(self, particle_i2c):
        self.graph = particle_i2c

    def build_plot(self, title_pre, fig_name=None):
        pass

    def plot_particle_forward_backwards_cells(self, dim, dim_name='x', fig=None):
        """This plots the (subsamled) forward and backward particle clouds
        for a given dimension with mean and sigma shaded.

        Arguments:
            dim {int} -- dimension to compute particles over

        Keyword Arguments:
            fig {plt} -- if fig is given, figure will be used and returned
            otherwise, cloud is printed directly to screen (default: {None})
        """
        if fig is None:
            # initialize new figure
            fig, axs = plt.subplots(1)

        f_particles = self.graph.all_f_samples[:, :, dim]
        b_particles = np.flip(self.graph.all_samples[:, :, dim], 0)
        
        time_x_loc_f = np.repeat(np.arange(self.graph.T), f_particles.shape[1])
        time_x_loc_b = np.repeat(np.arange(self.graph.T), b_particles.shape[1])

        if self.graph.is_multimodal:
            # TODO: Fix multimodal plotting to be nicer (via the gaussian comp sig)
            # TODO: Howto get prior after update? Rework code
            pass
        
        mean_f, sig_f, sig_upper_f, sig_lower_f = get_mean_sig_bounds(f_particles, 1, 2)
        mean_b, sig_b, sig_upper_b, sig_lower_b = get_mean_sig_bounds(b_particles, 1, 2)

        fig.fill_between(np.arange(self.graph.T), sig_lower_f, sig_upper_f, color='C0', alpha=.1)
        fig.fill_between(np.arange(self.graph.T), sig_lower_b, sig_upper_b, color='C1', alpha=.1)
        fig.plot(mean_f, color='C0')
        fig.plot(mean_b, color='C1')
        fig.scatter(time_x_loc_f, f_particles.flatten(), 0.01, color='C0')
        fig.scatter(time_x_loc_b, b_particles.flatten(), 0.01, color='C1')

        fig.set_xlabel('Timestep')
        fig.set_ylabel(dim_name + str(dim))
        fig.set_title('Forward/backward particles over ' + dim_name + str(dim))
        
        return fig

    def plot_joint_forward_backward_cells(self, dim, fig_name=None):
        pass

    def eval_controler(self, eval_env, cost, repeats=1000, random_starts=True):
        costs = []
        us = []
        x = eval_env.init_env(randomized=random_starts)
        print('Evaluating envs for plotting')
        for i in tqdm(range(repeats)):
            for i in range(self.graph.T):
                u = self.graph.get_policy(x.flatten(), i).reshape(-1,1)
                us.append(u)
                x = eval_env.forward(u)
                costs.append(cost(x.flatten(), u))
        costs = np.array(costs).reshape(repeats, -1)
        us = np.array(us).reshape(repeats, -1)
        return costs, us

    def plot_controler(self, eval_env, cost, repeats=1000, random_starts=True, fig=None):
        if fig is None:
            fig, ax = plt.subplots(2)
        else:
            ax = fig

        plt_help_x = np.arange(self.graph.T)
        costs, us = self.eval_controler(eval_env, cost, repeats=repeats, random_starts=random_starts)
        
        mean_c, sig_c, sig_upper_c, sig_lower_c = get_mean_sig_bounds(costs, 0, 2)
        mean_u, sig_u, sig_upper_u, sig_lower_u = get_mean_sig_bounds(us, 0, 2)
        # plot costs
        ax[0].fill_between(plt_help_x, sig_lower_c, sig_upper_c, color='C0', alpha=0.1)
        for i in range(repeats):
            ax[0].plot(costs[i], '--', color='C0', lw=0.5)
        ax[0].plot(mean_c, 'C0')
        ax[0].set_xlabel('T')
        ax[0].set_ylabel('cost')
        ax[0].set_title('Per timestep cost of several controler evaluations')

        # plot controls
        ax[1].fill_between(plt_help_x, sig_lower_u, sig_upper_u, color='C1', alpha=0.1)
        for i in range(repeats):
            ax[1].plot(us[i], '--', color='C1', lw=0.5)
        ax[1].plot(mean_u, 'C1')
        ax[1].set_xlabel('T')
        ax[1].set_ylabel('u')
        ax[1].set_title('Per timestep control signal of several controler evaluations')

        return fig, ax


    def plot_all(self, run_name, eval_env, cost, repeats=10, random_starts=True):
        plt.clf()
        fig, axs = plt.subplots(3)
        fig.suptitle('Particle I2C training ' + run_name)
        self.plot_particle_forward_backwards_cells(0, fig=axs[0])
        self.plot_particle_forward_backwards_cells(1, fig=axs[1])
        self.plot_particle_forward_backwards_cells(2, 'u',fig=axs[2])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # fixes for suptitle
        fig.savefig('plots/particles_{}.png'.format(run_name))
        fig, axs = plt.subplots(2)
        fig.suptitle('Particle I2C controler evaluation ' + run_name)
        self.plot_controler(eval_env, cost, repeats, random_starts, fig=axs)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # fixes for uptitle
        fig.savefig('plots/controler_{}.png'.format(run_name))


def get_mean_sig_bounds(arr, dim, sig_multiple=1.):
    mean_arr = arr.mean(dim)
    sig_arr = arr.std(dim)
    sig_upper_arr = mean_arr + sig_arr * sig_multiple
    sig_lower_arr = mean_arr - sig_arr * sig_multiple
    return mean_arr, sig_arr, sig_upper_arr, sig_lower_arr

import os
import jax
import jax.numpy as np
import torch
from tqdm import tqdm

from distutils.spawn import find_executable
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['figure.figsize'] = [19, 19]
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 22
matplotlib.rcParams['axes.labelsize'] = 22
if find_executable("latex"):
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt


class ParticlePlotter():

    def __init__(self, particle_i2c, config):
        self.graph = particle_i2c
        self.log_dir = config.LOGGING.log_dir
        if not os.path.exists('plots/' + self.log_dir):
            os.makedirs('plots/' + self.log_dir)
        self.policy_type = config.POLICY.type
        self.init_state_var = config.ENVIRONMENT.init_state_var
        self.i = 0

    def clean(self, arr):
        if self.policy_type == 'VSMC':
            # don't ask
            arr = [[a.detach().numpy() for a in ar] for ar in arr]
        arr = np.concatenate(np.stack([np.stack(a) for a in arr]), 1)
        return arr

    def build_plot(self, title_pre, fig_name=None):
        pass

    def plot_particle_forward_backwards_cells(self, f_particles, b_particles, weights, dim, dim_name='x', fig=None):
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
        
        f_particles = f_particles[..., dim]
        b_particles = b_particles[..., dim]

        b_weighing = jax.nn.softmax(weights)
        b_weighing *= (1/np.max(b_weighing, 1, keepdims=True))

        time_x_loc_f = np.repeat(np.arange(self.graph.T), f_particles.shape[1])
        time_x_loc_b = np.repeat(np.arange(self.graph.T), b_particles.shape[1])

        # if self.graph.is_multimodal:
        #     # TODO: Fix multimodal plotting to be nicer (via the gaussian comp sig)
        #     # TODO: Howto get prior after update? Rework code
        #     pass
        
        mean_f, sig_f, sig_upper_f, sig_lower_f = get_mean_sig_bounds(f_particles, 1, np.ones_like(b_weighing), 2)
        mean_b, sig_b, sig_upper_b, sig_lower_b = get_mean_sig_bounds(b_particles, 1, b_weighing, 2)

        fig.fill_between(np.arange(self.graph.T), sig_lower_f, sig_upper_f, color='C0', alpha=.05)
        fig.fill_between(np.arange(self.graph.T), sig_lower_b, sig_upper_b, color='C1', alpha=.1)
        fig.plot(mean_f, color='C0')
        fig.plot(mean_b, color='C1')
        fig.scatter(time_x_loc_f, f_particles.flatten(), 0.01, color='C0')
        fig.scatter(time_x_loc_b, b_particles.flatten(), b_weighing, color='C1')

        fig.set_xlabel('Timestep')
        fig.set_ylabel(dim_name + str(dim))
        fig.set_title('Forward/backward particles over ' + dim_name + str(dim))
        
        return fig

    def plot_joint_forward_backward_cells(self, dim, fig_name=None):
        pass

    def eval_controler(self, eval_env, cost, repeats=1000, random_starts=True):
        costs = []
        us = []
        print('Evaluating envs for plotting')
        for i in tqdm(range(repeats)):
            x = eval_env.init_env(self.init_state_var, randomized=random_starts)
            for i in range(self.graph.T):
                u = self.graph.get_policy(x.reshape(1,-1), i).reshape(1,1)
                us.append(u)
                costs.append(cost(x.reshape(1,-1), u))
                x = eval_env.sample(x, u)
        if self.policy_type == 'VSMC':
            costs = torch.tensor(costs).detach().numpy().reshape(repeats, -1)
            us = torch.tensor(us).detach().numpy().reshape(repeats, -1)
        else:
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
        
        mean_c, sig_c, sig_upper_c, sig_lower_c = get_mean_sig_bounds(costs, 0, np.ones_like(costs), 1)
        mean_u, sig_u, sig_upper_u, sig_lower_u = get_mean_sig_bounds(us, 0, np.ones_like(us), 1)
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

    def plot_all(self, alpha, f_particles, b_particles, weights, run_name, eval_env, cost, repeats=10, random_starts=True):
        f_particles = self.clean(f_particles)
        b_particles = np.flip(self.clean(b_particles), 0)
        weights = np.flip(self.clean(weights), 0)

        plt.clf()
        fig, axs = plt.subplots(3)
        fig.suptitle(f'Particle I2C training {run_name} {self.i}')
        for i in range(2):
            self.plot_particle_forward_backwards_cells(f_particles, b_particles, weights, i, fig=axs[i])
        self.plot_particle_forward_backwards_cells(f_particles, b_particles, weights, 2, 'u',fig=axs[2])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # fixes for suptitle
        fig.savefig(f'plots/{self.log_dir}/particles_{run_name}_{self.i}.png')
        fig, axs = plt.subplots(2)
        fig.suptitle(f'Particle I2C controler evaluation {run_name} {self.i}')
        self.plot_controler(eval_env, cost, repeats, random_starts, fig=axs)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # fixes for uptitle
        fig.savefig(f'plots/{self.log_dir}/controler_{run_name}_{self.i}.png')
        plt.close('all')
        self.i += 1


def get_mean_sig_bounds(arr, dim, weights, sig_multiple=1.):
    mean_arr = np.expand_dims(np.average(arr, dim, weights), dim)
    sig_arr = np.sqrt(np.mean(weights * (arr - mean_arr) ** 2, dim))
    mean_arr = mean_arr.squeeze()
    sig_upper_arr = mean_arr + sig_arr * sig_multiple
    sig_lower_arr = mean_arr - sig_arr * sig_multiple
    return mean_arr, sig_arr, sig_upper_arr, sig_lower_arr

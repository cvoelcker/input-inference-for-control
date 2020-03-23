from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as normal_density

from kalman import KalmanFilter


class ParticleFilter():

    def __init__(self, num_particles, initial_particles, measurement_likelihood, system):
        self.particles = initial_particles
        self.norm_weights = np.ones(num_particles) / num_particles
        self.weights = np.ones(num_particles) / num_particles
        self.num_p = num_particles
        self.m = measurement_likelihood
        self.f = system

        self.particles_save = [self.particles.copy()]
        self.weights_save = [self.weights.copy()]
        self.smoothing_weights = []

    def position_update(self, actuation):
        self.particles = self.f.sample(self.particles, actuation)
        self.particles_save.append(self.particles.copy())

    def reweighing(self, measurement):
        self.weights = self.m(self.particles, measurement)
        if not np.isclose(np.sum(self.weights), 1.):
            self.weights = self.weights/np.sum(self.weights)
        self.weights_save.append(self.weights.copy())

    def position_estimate(self):
        weighted_particles = self.particles * self.weights
        return np.sum(weighted_particles, 1), np.var(weighted_particles, 1)

    def resample(self):
        '''
        This implements multinomial resampling for now
        '''
        samples = np.random.choice(np.arange(self.num_p), size=self.num_p, replace=True, p=self.weights)
        self.particles = self.particles[:, samples]
        self.weights = np.ones(self.num_p) / self.num_p

    def filter(self, measurement, actuation=None):
        self.position_update(actuation)
        self.reweighing(measurement)
        pos = self.position_estimate()
        self.resample()
        return pos

    def filter_sequence(self, measurements, actuations=None):
        T = len(measurements)
        
        estimates_forward = []
        for t in range(T):
            pos = self.filter(measurements[t], actuations[t])
            estimates_forward.append(pos)

        return np.array(estimates_forward)

    def smooth_sequence(self, M):
        T = len(self.particles_save) - 1
        swarm = particles_save[T]

        trajectory_indices = np.random.choice(np.arange(M), size=M, replace=True, p=self.weights_save[T])
        trajectories = swarm[:, trajectory_indices]

        for t in range(T-1, -1, -1):
            self.f.likelihood
        


class System(ABC):

    @abstractmethod
    def sample(self, conditional=None):
        pass

    @abstractmethod
    def likelihood(self, observation, conditional=None):
        pass

    @abstractmethod
    def sample_observation(self, conditional):
        pass

    @abstractmethod
    def observation_likelihood(self, observation, estimate, alpha=1.):
        pass

    @abstractmethod
    def generate_trajectory(self, T):
        pass


class LQSystem(System):

    def __init__(self, A, B, Q, R, e, r, mu_0, sigma_0):
        '''
        A: self evolution
        B: actuation
        e: process noise
        Q: state observation
        R: actuation observation
        '''
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.e = e
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0

        # merge actuation and state into observation
        self.dim_x = self.A.shape[1]
        self.dim_u = self.B.shape[1]
        self.dim_ux = self.dim_x + self.dim_u

        AB = np.concatenate([self.A, self.B], 1)
        self.sys_linear = np.eye(self.dim_ux)
        self.sys_linear[:self.dim_x,:] = AB

        self.obs_linear = self.eye(self.dim_ux)
        self.obs_linear[:self.dim_x, :self.dim_x] = Q
        self.obs_linear[self.dim_x:, self.dim_x:] = R

    def reset(self):
        pass

    def sample(self, conditional=None):
        if conditional is None:
            sample = np.random.multivariate_normal(self.mu_0, self.sigma_0, 1).T
        else:
            sample = self.sys_linear.dot(conditional) 
            


def test():
    m = lambda x: np.array(x)
    a = m([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])
    u = m([[0.005, 0, 1, 0], [0, 0.005, 0, 1]]).T
    q = 0.1 * np.eye(4)
    # q[2,2] = 0.001
    # q[3,3] = 0.001
    h = 1. * np.eye(4)
    r = 1. * np.eye(4)
    p_init = 1. * np.eye(4)
    x_init = m([[0], [0], [1], [1]])

    num_particles = 100
    particles_init = np.random.uniform(-1, 1, size=(4, num_particles))
    particles_init[0,:] = 0
    particles_init[1,:] = 0

    running_x = x_init

    def system(running_x, actuation):
        sample = np.random.multivariate_normal(np.zeros(4), q, 1).T
        running_x = a.dot(running_x) + u.dot(actuation) + sample
        return running_x

    def proposal(running_x, actuation):
        sample = np.random.multivariate_normal(np.zeros(4), q, num_particles).T
        running_x = a.dot(running_x) + u.dot(actuation) + sample
        return running_x
    
    def sensor(position):
        sample = np.random.multivariate_normal(np.zeros(4), r, 1).T
        measure = h.dot(position) + sample
        return measure

    def sensor_likelihood(position, measurement, normalization=10.):
        likelihoods = []
        all_dist = (position - measurement.reshape(-1,1))
        weights = 1./((normalization * all_dist) ** 2).mean(0)
        # for i in range(position.shape[1]):
        #     l = normal_density.pdf(measurement.reshape(-1), position[:, i], normalization)
        #     likelihoods.append(l)
        return weights

    f = ParticleFilter(num_particles, particles_init, sensor_likelihood, proposal)
    
    all_x = []
    all_y = []
    all_sensor_x = []
    all_sensor_y = []
    all_particle_x = []
    all_particle_y = []
    noiseless_x = []
    noiseless_y = []
    naive_pos = x_init
    particles_x = []
    particles_y = []
    all_kalman_x = []
    all_kalman_y = []

    kalman = KalmanFilter(a, u, q, h, r)
    kalman.initialize(x_init, p_init)

    errors_particle = []
    errors_kalman = []

    for _ in range(1000):
        print(_)
        actuation = m([[1], [0]])
        running_x = system(running_x, actuation)
        measurement = sensor(running_x)

        pos, _ = f.filter(actuation, measurement)
        # particles_x.append(f.particles[0].copy())
        # particles_y.append(f.particles[1].copy())
        
        all_x.append(running_x[0])
        all_y.append(running_x[1])
        all_sensor_x.append(measurement[0])
        all_sensor_y.append(measurement[1])
        all_particle_x.append(pos[0])
        all_particle_y.append(pos[1])
        naive_pos = a.dot(naive_pos) + u.dot(m([[1], [0]]))
        noiseless_x.append(naive_pos[0])
        noiseless_y.append(naive_pos[1])
        
        kalman.time_update(actuation)
        pos_kalman, p = kalman.measure_update(measurement)
        all_kalman_x.append(pos_kalman[0])
        all_kalman_y.append(pos_kalman[1])

        errors_kalman.append(((running_x - pos_kalman) ** 2).mean())
        errors_particle.append(((running_x - pos.reshape(4,1)) ** 2).mean())
    plt.scatter(all_x, all_y)
    plt.scatter(all_particle_x, all_particle_y, alpha=0.8)
    plt.scatter(all_sensor_x, all_sensor_y, alpha=0.5)
    plt.scatter(all_kalman_x, all_kalman_y, alpha=0.8)
    # plt.scatter(noiseless_x, noiseless_y, 0.2)
    plt.scatter(m(particles_x).reshape(-1), m(particles_y).reshape(-1), 0.5, alpha=0.3)
    plt.show()

    plt.plot(errors_kalman)
    plt.plot(errors_particle)
    plt.show()

    print(np.array(errors_kalman).mean())
    print(np.array(errors_particle).mean())

    return True


if __name__=="__main__":
    print(test())

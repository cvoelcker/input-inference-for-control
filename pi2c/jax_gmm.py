from tqdm import tqdm

import jax.numpy as np
import numpy
import jax
from jax import grad, jit, vmap, value_and_grad, random
import matplotlib.pyplot as plt


class CycleKey():
    KEY = random.PRNGKey(0)

    def __call__(self):
        CycleKey.KEY, sk = random.split(CycleKey.KEY)
        return sk

seed = CycleKey()


@jit
def softmax(vec):
    return np.exp(vec)/np.sum(np.exp(vec))


@jit
def gaussian_pdf(mu, var, x):
    dim = mu.size
    norm = ((2 * np.pi) ** (-dim/2)) * (np.linalg.det(var) ** (-1/2))
    return norm * np.exp((-1/2) * (x - mu).T @ np.linalg.inv(var) @ (x - mu))

@jit
def log_normal_pdf(mu, var, x):
    dim = mu.size
    x = x.reshape(1,-1)
    norm = np.log(((2 * np.pi) ** (-dim/2)) * (np.linalg.det(var) ** (-1/2)))
    return norm + (-1/2) * (x - mu) @ np.linalg.inv(var) @ (x - mu).T


vec_log_normal_pdf = jit(vmap(log_normal_pdf, in_axes=(None, None, 0), out_axes=0))


@jit
def gmm(params, x):
    pi = params[0]
    mu = params[1]
    var = params[2]
    return np.sum(pi * vmap(gaussian_pdf, in_axes=(0,0,None))(mu, var, x))


@jit
def gmm_softmax_pi(params, x):
    pi = params[0]
    pi = np.exp(pi)/np.sum(np.exp(pi))
    mu = params[1]
    var = params[2]
    return gmm((pi, mu, var), x)


def gmm_condition(params, x, idx):
    pi = params[0]
    mu = params[1]
    var = params[2]

    # x = x.reshape(1, -1)

    weight_func = lambda _x: vmap(gaussian_pdf, in_axes=(0,0,None), out_axes=0)(mu[:, :idx], var[:, :idx, :idx], _x)
    # reweight = vmap(weight_func)(x)
    reweight = weight_func(x)

    def var_mu(_var, _mv, _mo):
        v_corr = _var[idx:, :idx] @ np.linalg.inv(_var[:idx, :idx])
        _mu = _mv + v_corr @ (x - _mo)
        _var_new = _var[idx:, idx:] - v_corr @ _var[:idx, idx:]
        return _mu, _var_new
    new_mu, new_var = vmap(var_mu)(var, mu[:, idx:], mu[:, :idx])
    return reweight, new_mu, new_var


def mean_grad(f, p, x):
    def mean_f(p, x):
        return np.mean(f(p, x))
    return value_and_grad(mean_f)(p, x)


def weighted_mean_grad(f, p, x, weights):
    def mean_f(p, x):
        return np.mean(f(p, x) * weights)
    return value_and_grad(mean_f)(p, x)


@jit
def empirical_cov(x, mu, weights=None):
    def cov_i(_x):
        _x = _x.reshape(1,-1)
        return ((_x - mu).T @ (_x - mu))
    if weights is None:
        return np.sum(vmap(cov_i)(x),0)/(x.shape[0]-1)
    else:
        return np.sum(vmap(cov_i)(x) * weights.reshape(-1,1,1),0)/np.sum(weights)


@jit
def empirical_mu(x, weights=None):
    if weights is None:
        return np.mean(x,0)
    else:
        return np.sum(x * weights,0)/weights.sum()


class GMM:

    def __init__(self, dim, n_components, idx, sig0=10000., key=0):
        self._pi = np.ones(n_components) / n_components
        self._mu = random.normal(seed(), (n_components, dim)) * 3.
        self._var = np.eye(dim).reshape(1,dim,dim).repeat(n_components,0) * sig0
        self._sig = vmap(np.linalg.cholesky)(self._var)

        self.n_components = n_components
        self.dim = dim
        self.idx = idx

    @property
    def params(self):
        return (self._pi, self._mu, self._var)

    def __call__(self, x, n):
        return self.conditional_sample(x, self.idx, n)

    def likelihood(self, x):
        return vmap(gmm, in_axes=(None,0), out_axes=0)(self.params, x)

    def conditional_likelihood(self, x, u):
        joint_x = np.concatenate([x, u], -1)
        joint_ll = self.likelihood(joint_x)
        marginal_ll = vmap(gmm, in_axes=(None,0), out_axes=0)(
            (self._pi, self._mu[:, :self.idx], self._var[:, :self.idx, :self.idx]), x)
        return joint_ll/marginal_ll

    def log_likelihood(self, x):
        return np.log(self.likelihood(x))

    def sample(self, n):
        comp = random.categorical(
                seed(),
                np.log(self._pi),
                shape=(n,))
        ran = random.normal(seed(), (n, 1, self.dim))
        samples = self._mu[comp].reshape(n, 1, -1) + ran @ self._sig[comp]
        return samples

    def mean(self):
        return np.sum(self._mu * self._pi.reshape(-1,1), 0)

    def condition(self, x, idx):
        _gmm_condition = lambda _x: gmm_condition(self.params, _x, idx)
        pi, mu, var = vmap(_gmm_condition)(x)
        return pi, mu, var

    def conditional_sample(self, x, idx, n):
        reps = x.shape[0]
        _idx_help = np.arange(n*reps)
        pi, mu, var = self.condition(x, idx)
        var = np.maximum(var, 1e-5)
        sig = vmap(np.linalg.cholesky)(var)

        pi = np.repeat(pi, n, 0)
        mu = np.repeat(mu, n, 0)
        sig = np.repeat(sig, n, 0)
        sig = np.maximum(sig, 0.)
        comp = random.categorical(
                seed(),
                np.log(pi),
                axis=1).reshape(-1)
        offset = mu[_idx_help, comp]
        ran = random.normal(seed(), (n*reps, 1, self.dim-idx))
        ran = (ran @ sig[_idx_help, comp]).reshape(-1, mu.shape[-1])
        samples = offset + ran
        return samples

    def conditional_mean(self, x, idx):
        x = x.reshape(-1, idx)
        pi, mu, var = self.condition(x, idx)
        if not np.isclose(np.sum(pi), 1.):
            pi = self._pi
        weighted_mean = np.sum(pi * mu.reshape(x.shape[0], -1), 1)
        return weighted_mean

    # def gradient_update(self, x, alpha=1e-2):
    #     weight_func = lambda _x: vmap(gaussian_pdf, in_axes=(0,0,None), out_axes=0)(self._mu, self._var, _x)
    #     weights = vmap(weight_func)(x)
    #     weights = (weights/np.sum(weights,1).reshape(-1,1))
    #     #n_cov = np.stack([empirical_cov(x, self._mu[i].reshape(1,-1), weights[:,i]) for i in range(self.n_components)], 0)
    #     
    #     ll = lambda _p, _x: np.log(vmap(gmm_softmax_pi, in_axes=(None,0), out_axes=0)(_p, _x))
    #     value, grads = mean_grad(ll, self.params, x)
    #     grad_pi = grads[0]
    #     grad_mu = grads[1]
    #     grad_var = grads[2]
    #     
    #     self._pi = weights.sum(0)/weights.sum()
    #     self._mu += alpha * grad_mu
    #     self._var += alpha * grad_var
    #     #self._var = (1-alpha) * self._var + alpha * n_cov

    #     return grads

    def _smoothed_avg(self, x0, x1, alpha):
        return (1-alpha) * x0 + alpha * x1

    def update_parameters(self, x, weights, alpha=1., max_iters=3):
        assert not np.any(np.isnan(x))
        converged = False
        iters = 0
        while not converged:
            converged = self.em_update(x, weights, alpha)
            if iters == max_iters:
                break
            if iters >= 0:
                iters += 1

    def em_update(self, x, particle_weights, alpha=5e-2):
        assert not np.any(np.isnan(x))
        weight_func = lambda _x: vmap(gaussian_pdf, in_axes=(0,0,None), out_axes=0)(self._mu, self._var, _x)
        weights = vmap(weight_func)(x)
        weights += 1e-20
        # print(weights)
        weights = (weights/np.sum(weights,1).reshape(-1,1))
        mu = np.stack(
            [empirical_mu(x, (weights[:,i] * np.exp(particle_weights)).reshape(-1,1)) 
            for i in range(self.n_components)], 0)
        n_cov = np.stack(
            [empirical_cov(x, mu[i].reshape(1,-1), weights[:,i] * np.exp(particle_weights).reshape(-1)) 
            for i in range(self.n_components)], 0)
        converged = np.all(np.isclose(self._pi, weights.sum(0)/weights.sum())) \
                and np.all(np.isclose(self._mu, mu)) \
                and np.all(np.isclose(self._var, n_cov))
        weights *= np.exp(particle_weights).reshape(-1,1)
        self._pi = self._smoothed_avg(self._pi, weights.sum(0)/weights.sum(), alpha)
        self._mu =  self._smoothed_avg(self._mu, mu, alpha)
        self._var = self._smoothed_avg(self._var, n_cov, alpha)
        # print(self._var)
        assert not np.any(np.isnan(self._mu))
        return converged

if __name__ == "__main__":

    key = random.PRNGKey(0)
    
    pi = np.zeros(4)
    mu = np.ones((4, 2))
    var = np.eye(2).reshape(1,2,2).repeat(4,0)

    x = np.array([
            [0., 0.],
            [1., 1.],
            [2., 2.]])

    params = (pi, mu, var)

    my_gmm = GradientGMM(2, 3, 5)
    my_gmm(x)

    cov1 = numpy.array([[0.2, 0.0], [0.0, 0.2]])
    cov2 = numpy.array([[0.5, 0.1], [0.1, 0.5]])
    cov2 = numpy.array([[0.5, 0.2], [0.2, 0.1]])
    cov3 = numpy.array([[0.5, 0.2], [0.2, 0.1]])
    
    mu1 = numpy.array([2., 2.])
    mu2 = numpy.array([0., 0.])
    mu3 = numpy.array([-2., -2.])
    
    samples1 = numpy.random.multivariate_normal(mu1, cov1, 3000)
    samples2 = numpy.random.multivariate_normal(mu2, cov2, 1000)
    samples3 = numpy.random.multivariate_normal(mu3, cov3, 2000)

    samples = np.concatenate([samples1, samples2, samples3], 0)

    alpha = 5e-2
    for i in tqdm(range(1000)):
        # _samples = samples[numpy.random.choice(6000, 1024, replace=False)]
        # my_gmm.update(_samples, alpha)
        
        my_gmm.condition(samples2[:, :1], 1)
        my_gmm.sample(100)
        my_gmm.conditional_sample(samples2[:,:1], 1, 10)
        my_gmm.conditional_mean(samples2[:,:1], 1)


        # alpha = 0.99 * alpha

        # plot_2d(my_gmm)
        # plot_scatter(samples)
        # plt.pause(.01)
        # plt.clf()
        # plt.show()

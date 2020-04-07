import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from pi2c.utils import GMM

def plot_scatter(points):
    plt.scatter(points[:, 0], points[:, 1], 0.2)

def plot_2d(f):
    x, y = np.mgrid[-10:10:.05, -10:10:.05]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    plt.contourf(x, y, f(pos))

mu_init = np.array([
    [0.0, 0.0],
    [1., 1.],
    [-1., -1.],
])
pi_init = np.array([1., 1., 1.])/3
sig_init = 1.

cov1 = np.array([[0.2, 0.0], [0.0, 0.2]])
cov2 = np.array([[0.5, 0.1], [0.1, 0.5]])
cov2 = np.array([[0.5, 0.2], [0.2, 0.1]])
cov3 = np.array([[0.5, 0.2], [0.2, 0.1]])

mu1 = np.array([1., 1.])
mu2 = np.array([0., 0.])
mu3 = np.array([-1., -1.])

samples1 = np.random.multivariate_normal(mu1, cov1, 300)
samples2 = np.random.multivariate_normal(mu2, cov2, 100)
samples3 = np.random.multivariate_normal(mu3, cov3, 200)

samples = np.concatenate([samples1, samples2, samples3], 0)

gmm = GMM(mu_init, pi_init, sig_init)

gmm.update_parameters(samples)

plot_2d(gmm)
plot_scatter(samples1)
plot_scatter(samples2)
plot_scatter(samples3)
# plot_scatter(gmm.sample(10000))

plt.show()
plot_2d(gmm)
plt.show()
print(gmm.sig)
print(cov1)
print(cov2)
print(cov3)
print(gmm.pi)

print('converged')

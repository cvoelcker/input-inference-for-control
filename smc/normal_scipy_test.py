from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

x, y = np.mgrid[-10:10:.05, -10:10:.05]
cov3 = np.array([[0.5, 0.2], [0.2, 0.1]])
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([1.0, 1.0], cov3)
plt.contourf(x, y, rv.pdf(pos))

samples = rv.rvs(1000)
def plot_scatter(points):
    plt.scatter(points[:, 0], points[:, 1], 0.2)
plot_scatter(samples)
plt.show()
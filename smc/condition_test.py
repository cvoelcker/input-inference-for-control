import numpy as np
import matplotlib.pyplot as plt
from pi2c.utils import GMM

def plot_scatter(points):
    plt.scatter(points[:, 0], points[:, 1], 0.2)

def plot_2d(f):
    x, y = np.mgrid[-10:10:.05, -10:10:.05]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    plt.contourf(x, y, f(pos))

mu1 = np.array([0.,  0., 5.])
mu2 = np.array([0., 1., 0.])
mu3 = np.array([2.,  0., -8])

gmm = GMM(np.array([mu1, mu2, mu3]), np.array([1.]*3)/3., 1.)

gmm_2 = gmm.condition(np.array([0., 0.]), 2)
x = np.linspace(-15,15,1000)
y = gmm_2(x)

plt.plot(x,y)

#plot_2d(gmm_2)
plt.show()

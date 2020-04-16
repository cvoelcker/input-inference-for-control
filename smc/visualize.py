import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

def plot_scatter(points):
    plt.scatter(points[:, 0], points[:, 1], 0.2)

def plot_2d(f):
    x, y = np.mgrid[-20:20:.2, -100:100:.2]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    plt.contourf(x, y, f(pos))

l = []

for i in range(100):
    l.append(np.load('results/back_{}.npy'.format(i)))

l = np.array(l)

# for i in range(100):
#     plt.scatter([i] * l.shape[1], l[i,:,2], 0.1)
#     print(l[i,:,0].mean())
# plt.show()

for i in reversed(range(100)):
    sig = np.load('results/sig_{}.npy'.format(i))[0]
    print(sig)
    var = sig @ sig.T
    print(var)
    gaussian = multivariate_normal(np.load('results/mu_{}.npy'.format(i))[0,1:], var[1:,1:])
    plot_2d(gaussian.pdf)
    plt.scatter(l[i,:, 1], l[i,:, 2], 0.1)
    plt.title('{}'.format(i))
    plt.show()

import numpy as np

from pi2c.utils import GMM

mu = np.array(
        [[0., 0., 0.],
         [1., 1., 1.],
         [2., 2., 2.],
         [3., 3., 3.],
         [4., 4., 4.]])

sig_scale = 0.1
weights = np.array([1., 1., 1., 1., 1.])/5.

test = GMM(mu, weights, sig_scale)

print(test)

print(test.conditional_mean(np.array([0., 0.]), [0, 1]))

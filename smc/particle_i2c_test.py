from pi2c.particle_i2c import QRCost, StaticQRCost, Cost2Prob
import numpy as np
Q = np.eye(2)
R = np.eye(1)
x = np.array([[1.], [1.]])
u = np.array([[0.]])
cost = StaticQRCost(Q, R, np.array([[0.], [0.]]), np.array([[0.]]))
print(cost.cost(x, u))
prob = Cost2Prob(cost)
print(prob.likelihood(x, u))

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter():

    def __init__(self, system_model, system_actuation, system_noise, sensor_model, sensor_noise):
        self.a = system_model
        self.b = system_actuation
        self.q = system_noise
        self.h = sensor_model
        self.r = sensor_noise

    def initialize(self, x_init, p_init):
        self.x = x_init
        self.p = p_init

    def time_update(self, u):
        x = self.x
        self.x_prop = self.a.dot(x) + self.b.dot(u)
        self.p_prop = self.a.dot(self.p).dot(self.a.T) + self.q
        return self.x_prop, self.p_prop

    def measure_update(self, measurement, x_prop=None, p_prop=None):
        p = self.p_prop if p_prop is None else p_prop
        x = self.x_prop if x_prop is None else x_prop
        measure_expected = self.h.dot(x)
        inv = np.linalg.inv(self.h.dot(p).dot(self.h.T) + self.r)
        k = p.dot(self.h.T).dot(inv)
        self.x = x + k.dot(measurement - measure_expected)
        self.p = (np.eye(p.shape[-1]) - k.dot(self.h)).dot(p)
        return self.x, self.p


def test():
    m = lambda x: np.array(x)
    a = m([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])
    u = m([[0.005, 0, 1, 0], [0, 0.005, 0, 1]]).T
    q = 1 * np.eye(4)
    q[2,2] = 0.001
    q[3,3] = 0.001
    h = np.eye(4)[:2]
    r = 1000 * np.eye(2)
    p_init = 100 * np.eye(4)
    x_init = m([[0], [0], [0], [0]])

    running_x = x_init

    def system(running_x, actuation):
        sample = np.random.multivariate_normal(np.zeros(4), q, 1).T
        running_x = a.dot(running_x) + u.dot(actuation) + sample
        sample = np.random.multivariate_normal(np.zeros(2), r, 1).T
        sensor = h.dot(running_x) + sample
        return running_x, sensor

    f = KalmanFilter(a, u, q, h, r)
    f.initialize(x_init, p_init)
    
    all_x = []
    all_y = []
    all_sensor_x = []
    all_sensor_y = []
    all_kalman_x = []
    all_kalman_y = []
    noiseless_x = []
    noiseless_y = []
    naive_pos = x_init

    for _ in range(5000):
        running_x, sensor = system(running_x, m([[1], [0]]))
        f.time_update(m([[1], [0]]))
        pos_kalman, p = f.measure_update(sensor)
        print(p)
        all_x.append(running_x[0])
        all_y.append(running_x[1])
        all_sensor_x.append(sensor[0])
        all_sensor_y.append(sensor[1])
        all_kalman_x.append(pos_kalman[0])
        all_kalman_y.append(pos_kalman[1])
        naive_pos = a.dot(naive_pos) + u.dot(m([[1], [0]]))
        noiseless_x.append(naive_pos[0])
        noiseless_y.append(naive_pos[1])
    plt.scatter(all_x, all_y, 0.2)
    plt.scatter(all_sensor_x, all_sensor_y, 0.2)
    plt.scatter(all_kalman_x, all_kalman_y, 0.2)
    # plt.scatter(noiseless_x, noiseless_y, 0.2)
    plt.show()

    return True


if __name__=="__main__":
    print(test())

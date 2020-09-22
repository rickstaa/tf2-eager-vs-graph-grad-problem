import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt
# This example is the RL based stationary Kalman filter


class Ex4_EKF(gym.Env):

    def __init__(self):

        self.t = 0
        self.dt = 0.1
        self.q1 = 0.01
        self.g=9.81

        self.mean0 = [1.5, 0]
        self.cov0_1 = 0.1
        self.cov0_2 = 0.1
        # self.cov0_1 = 0
        # self.cov0_2 = 0
        self.mean1 = [0,0]
        self.cov1= np.array([[1 / 3 * (self.dt) ** 3 * self.q1, 1 / 2 * (self.dt) ** 2 * self.q1], \
                              [1 / 2 * (self.dt) ** 2 * self.q1, self.dt * self.q1]])
        # self.cov1 = np.array([[0,0],[0,0]])
        self.mean2 = 0
        self.cov2 = 1e-2
        # self.cov2 = 0

        self.sigma = 0
        # displacement limit set to be [-high, high]
        high = np.array([10000,10000])

        self.action_space = spaces.Box(low=np.array([-10.,-10.,-10.,-10. ]), high=np.array([10.,10.,10.,10.]),dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.output = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, sysid):
        u1, u2 = action
        u3, u4 = sysid


        t = self.t
        input = 0*np.cos(t) * self.dt
        # Slave
        hat_x_1, hat_x_2, x_1, x_2 = self.state
        y_1 = self.output
        hat_y_1 = np.sin(hat_x_1)

        # hat_x_1 = self.dt * u1*(y_1-hat_y_1) + u3
        # hat_x_2 = self.dt * u2*(y_1-hat_y_1) + u4 + input

        # u3 = x_2
        # u4 = - self.g * np.sin(x_1)

        hat_x_1 = hat_x_1 + self.dt * u3 + self.dt * u1 * (y_1 - hat_y_1)
        hat_x_2 = hat_x_2 + self.dt * u4 + self.dt * u2 * (y_1 - hat_y_1)  + input

        # hat_x_1 = hat_x_1 + self.dt * x_2 + self.dt * u1 * (y_1 - hat_y_1)
        # hat_x_2 = hat_x_2 - self.g * np.sin(x_1) * self.dt + self.dt * u2 * (y_1 - hat_y_1) + input

        # Master
        x_1 = x_1 + self.dt * x_2
        x_2 = x_2 - self.g * np.sin(x_1) * self.dt + input
        state = np.array([x_1, x_2])
        # add process noise
        state = state + np.random.multivariate_normal(self.mean1, self.cov1).flatten()
        x_1, x_2 = state

        # to construct cost
        hat_y_1 = np.sin(hat_x_1)
        y_1 = np.sin(x_1) + np.random.normal(self.mean2, np.sqrt(self.cov2))
        cost_u = np.square(u1*(y_1-hat_y_1))* self.dt + np.square(u2*(y_1-hat_y_1))* self.dt
        cost_y = np.abs(hat_y_1 - y_1) * self.dt
        cost = cost_y
        # cost = np.square(hat_x_1 - x_1) + np.square(hat_x_2 - x_2)
        # cost = np.abs(hat_x_1 - x_1)**1 + np.abs(hat_x_2 - x_2)**1
        # print('cost',cost)
        if cost > 100:
            done = True
        else:
            done = False


        # update new for next round
        self.state = np.array([hat_x_1, hat_x_2, x_1, x_2])
        self.output = y_1
        self.t = self.t + self.dt

        # return np.array([hat_x_1,hat_x_2,y_1, y_2]), cost, done, dict(reference=y_1, state_of_interest=np.array([hat_y_1,hat_y_2]))

        return np.array([hat_x_1, hat_x_2]), cost, done, dict(reference=y_1, state_of_interest=np.array([hat_x_1,x_1]))

    def reset(self):
        self.state = np.array([np.random.uniform(-3*np.pi/4,3*np.pi/4),np.random.uniform(-3*np.pi/4,3*np.pi/4),
                               np.random.uniform(-np.pi/2,np.pi/2),np.random.uniform(-np.pi/2,np.pi/2)])
        hat_x_1, hat_x_2, x_1, x_2 = self.state
        self.output = np.sin(x_1) + np.random.normal(self.mean2,np.sqrt(self.cov2))
        y_1 = self.output
        hat_y_1 = np.sin(hat_x_1)
        return np.array([hat_x_1,hat_x_2])

    def render(self, mode = 'human'):

        return

if __name__ == '__main__':
    env = Ex4_EKF()
    T = 10
    path = []
    t1 = []
    s = env.reset()
    for i in range(int(T/env.dt)):
        s, r, info, done = env.step(np.array([0.1, 0.2]),np.array([-0.4, 0.9]))
        path.append(s)
        t1.append(i * env.dt)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot(t1, path, color='blue', label='0.1')
    # ax.plot(t1, np.array(path)[:, 0], color='blue', label='Angle')
    # ax.plot(t1, np.array(path)[:, 1], color='green', label='Frequency')
    ax.plot(t1, np.array(path)[:, 0], color='yellow', label='x1')
    ax.plot(t1, np.array(path)[:, 1], color='green', label='x2')
    # ax.plot(t1, np.array(path)[:, 2], color='black', label='measurement')
    # ax.plot(t1, np.array(path)[:, 4], color='red', label='Output estimate error')

    handles, labels = ax.get_legend_handles_labels()
    #
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')







import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt

# the dynamic system whose state is to be estimated:
# x(k+1)=Ax(k)+Bw(k)
# x_1: Distance
# x_2: Speed
# y(k)=Cx(k)+v(k)
# A=[1,1;0,1]
# B=[0;1]
# C=[1,0]
# x(0)~N([0;10],[2,0;0,3])
# w(k)~N(0,1)
# v(k)~N(0,2)
class Ex2_EKF(gym.Env):

    def __init__(self):
        # assume that we know the system matrix A

        self.var0 = np.sqrt(0.5**2)
        self.var1 = np.sqrt(0.2**2)
        self.var2 = np.sqrt(0.2**2)

        self.t = 0
        self.dt = 1.
        self.sigma = 0
        # displacement limit set to be [-high, high]
        high = np.array([10000,10000])

        self.action_space = spaces.Box(low=np.array([-20]), high=np.array([20]),dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        u, = action
        hat_x, x = self.state
        # why need to add an artificial noise below ?
        # hat_x_1 = self.a11 * hat_x_1 + self.a12 * hat_x_2 + u1 + np.random.uniform(-self.sigma,self.sigma,1)
        # hat_x_2 = self.a21 * hat_x_1 + self.a22 * hat_x_2 + u2 + np.random.uniform(-self.sigma,self.sigma,1)

        hat_x = 0.5 * hat_x + u
        # hat_x = u
        hat_y = 5 * np.sin(hat_x)

        x = 0.5 * x + 25 * x / (1 + x ** 2) + np.random.normal(0,self.var1)
        y = 5 * np.sin(x) + np.random.normal(0,self.var2)
        self.state = np.array([hat_x,x])

        # r1 = self.reference(self.t)

        self.t = self.t + 1

        cost = np.abs(hat_y - y)**0.5
        # print('cost',cost)
        if cost > 100:
            done = True
        else:
            done = False
        return np.array([hat_x,y]), cost, done, dict(reference=y, state_of_interest=hat_x-x)

    def reset(self):
        self.state = np.array([-1+np.random.normal(0,self.var0),-1])
        hat_x, x = self.state
        y = 5 * np.sin(x) + np.random.normal(0,self.var2)
        return np.array([hat_x,y])

    def render(self, mode = 'human'):

        return

if __name__ == '__main__':
    env = Ex2_EKF()
    T = 1000
    path = []
    t1 = []
    s = env.reset()
    for i in range(int(T/env.dt)):
        s, r, info, done = env.step(np.array([0]))
        path.append(s)
        t1.append(i * env.dt)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot(t1, path, color='blue', label='0.1')
    ax.plot(t1, np.array(path)[:, 0], color='blue', label='State estimate')
    ax.plot(t1, np.array(path)[:, 1], color='green', label='Measurement')
    # ax.plot(t1, np.array(path)[:, 2], color='yellow', label='Distance')
    # ax.plot(t1, np.array(path)[:, 1], color='green', label='Speed estimate')
    # ax.plot(t1, np.array(path)[:, 3], color='black', label='Speed')
    # ax.plot(t1, np.array(path)[:, 4], color='red', label='Output estimate error')

    handles, labels = ax.get_legend_handles_labels()
    #
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')







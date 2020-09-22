import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt
# This example is the RL based stationary Kalman filter

# the dynamic system whose state is to be estimated:
# x(k+1)=Ax(k)+w(k)
# x_1: angle
# x_2: frequency
# x_3: amplitude
# y(k)=x_3(k)*sin(x_1(k))+v(k)
# A=[1,dt,0;0,1,0;0,0,1]
# x(0)~N([0;10;1],[3,0,0;0,3,0;0,0,3])
# w(k)~N([0;0;0],[1/3*(dt)^3*q_1,1/2*(dt)^2*q_1,0;1/2*(dt)^2*q_1,dt*q_1,0;0,0,dt*q_2])
# v(k)~N(0,1)

# estimator design
# \hat(x)(k+1)=A\hat(x)(k)+u
# where u=[u1,u2,u3]', u=l(\hat(x)(k),y(k)) come from the policy network l(.,.)

class Ex1_EKF(gym.Env):

    def __init__(self):

        self.t = 0
        self.dt = 0.1
        self.q1 = 0.002
        self.q2 = 0.001
        # the system matrix A
        self.A = np.array([[1,self.dt,0],[0,1,0],[0,0,1]])

        self.mean0 = [0,1,1]
        self.cov0 = 3e-2*np.eye(3)
        self.mean1 = [0,0,0]
        self.cov1 = np.array([[1 / 3 * (self.dt) ** 3 * self.q1, 1 / 2 * (self.dt) ** 2 * self.q1, 0], \
                              [1 / 2 * (self.dt) ** 2 * self.q1, self.dt * self.q1, 0], [0, 0, self.dt * self.q2]])
        self.mean2 = 0
        self.cov2 = 1e-2

        self.sigma = 0
        # displacement limit set to be [-high, high]
        high = np.array([10000,10000,10000,10000])

        self.action_space = spaces.Box(low=np.array([-20.,-20.,-20.]), high=np.array([20.,20.,20.]),dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        u1, u2, u3 = action
        hat_x_1, hat_x_2, hat_x_3, x_1, x_2, x_3 = self.state
        # why need to add an artificial noise below ?
        # hat_x_1 = self.a11 * hat_x_1 + self.a12 * hat_x_2 + u1 + np.random.uniform(-self.sigma,self.sigma,1)
        # hat_x_2 = self.a21 * hat_x_1 + self.a22 * hat_x_2 + u2 + np.random.uniform(-self.sigma,self.sigma,1)
        hat_x = np.array([[hat_x_1],[hat_x_2],[hat_x_3]])
        x = np.array([[x_1],[x_2],[x_3]])
        hat_x = np.dot(self.A,hat_x) + np.array([[u1],[u2],[u3]])
        hat_y = hat_x[2,0] * np.sin(hat_x[0,0])

        x = np.dot(self.A,x) + np.random.multivariate_normal(self.mean1,self.cov1).reshape((3,1))
        # x = np.dot(self.A, x)
        y = x[2,0] * np.sin(x[0,0]) + np.random.normal(self.mean2,self.cov2)
        self.state = np.concatenate((hat_x,x),axis=0).flatten()
        self.state.flatten()

        # r1 = self.reference(self.t)

        self.t = self.t + 1

        #  cost = abs(hat_y - y)**0.5
        cost = np.abs(hat_y - y)
        # print('cost',cost)
        if cost > 100:
            done = True
        else:
            done = False
        return np.array([hat_x[0,0],hat_x[1,0],hat_x[2,0],y]), cost, done, dict(reference=y, state_of_interest=hat_x[0,0]-x[0,0])

    def reset(self):
        self.state = np.concatenate((np.random.multivariate_normal(self.mean0,self.cov0),np.array(self.mean0)))
        hat_x_1, hat_x_2, hat_x_3, x_1, x_2, x_3 = self.state
        hat_y = hat_x_3 * np.sin(hat_x_1)
        y = x_3 * np.sin(x_1) + np.random.normal(self.mean2,self.cov2)
        return np.array([hat_x_1,hat_x_2,hat_x_3,y])

    def render(self, mode = 'human'):

        return

if __name__ == '__main__':
    env = Ex1_EKF()
    T = 40
    path = []
    t1 = []
    s = env.reset()
    for i in range(int(T/env.dt)):
        s, r, info, done = env.step(np.array([0, 0, 0]))
        path.append(s)
        t1.append(i * env.dt)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot(t1, path, color='blue', label='0.1')
    # ax.plot(t1, np.array(path)[:, 0], color='blue', label='Angle')
    # ax.plot(t1, np.array(path)[:, 1], color='green', label='Frequency')
    # ax.plot(t1, np.array(path)[:, 2], color='yellow', label='Amplitude')
    # ax.plot(t1, np.array(path)[:, 1], color='green', label='Speed estimate')
    ax.plot(t1, np.array(path)[:, 3], color='black', label='measurement')
    # ax.plot(t1, np.array(path)[:, 4], color='red', label='Output estimate error')

    handles, labels = ax.get_legend_handles_labels()
    #
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 21:28:15 2017

@author: Narifumi
"""
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random
clf()

# 訓練データ
N_blue = 100
N_red = 100
mu_blue = [1, 3]
mu_red = [3, 1]

cov = [[0.8, 0.0], [0.0, 0.1]]
x_blue = np.random.multivariate_normal(mu_blue, cov, N_blue).T
x_red = np.random.multivariate_normal(mu_red, cov, N_red).T


m1 = np.matrix(x_blue.sum(1) / N_blue).T
m2 = np.matrix(x_red.sum(1) / N_red).T

mid = (m2 + m1) / 2
w1 = m2 - m1
x = [-1, 0]
x[1] = float((w1[0] * mid[0] + w1[1] * mid[1]) / w1[1] - (w1[0] / w1[1]) * x[0])

plt.subplot(2, 2, 1)
plt.scatter(x_blue[0], x_blue[1], color='b', alpha=0.5, s=5.0)
plt.scatter(x_red[0], x_red[1], color='r', alpha=0.5, s=5.0)
plt.plot(m1[0, 0], m1[1, 0], 'b+')
plt.plot(m2[0, 0], m2[1, 0], 'r+')
plt.plot([m1[0, 0], m2[0, 0]], [m1[1, 0], m2[1, 0]], 'g-')
plt.plot([mid[0, 0], x[0]], [mid[1, 0], x[1]], 'g-')
plt.axis('equal')

plt.subplot(2, 2, 3)
plt.hist(array(w1.T.dot(x_blue))[0], color='b', bins=10, density=True)
plt.hist(array(w1.T.dot(x_red))[0], color='r', bins=10, density=True)

# フィッシャーの線形判別
Sw = np.zeros([2, 2])
for i in range(N_blue):
    Sw += (np.matrix(x_blue[:, i]).T - m1).dot((np.matrix(x_blue[:, i]).T - m1).T)
for i in range(N_red):
    Sw += (np.matrix(x_red[:, i]).T - m2).dot((np.matrix(x_red[:, i]).T - m2).T)

w2 = 20 * np.linalg.inv(Sw).dot(m2 - m1)
x[1] = float((w2[0] * mid[0] + w2[1] * mid[1]) / w2[1] - (w2[0] / w2[1]) * x[0])

plt.subplot(2, 2, 2)
plt.scatter(x_blue[0], x_blue[1], color='b', alpha=0.5, s=5.0)
plt.scatter(x_red[0], x_red[1], color='r', alpha=0.5, s=5.0)
plt.plot(m1[0, 0], m1[1, 0], 'b+')
plt.plot(m2[0, 0], m2[1, 0], 'r+')
plt.plot([float(0 + mid[0, 0] - w2[0, 0] / 2), float(w2[0] + mid[0, 0] - w2[0, 0] / 2)],
         [float(0 + mid[1, 0] - w2[1, 0] / 2), float(w2[1] + mid[1, 0] - w2[1, 0] / 2)], 'g-')

# plt.plot([m1[0,0],m2[0,0]],[m1[1,0],m2[1,0]],'g-')
plt.plot([mid[0, 0], x[0]], [mid[1, 0], x[1]], 'g-')
plt.axis('equal')

plt.subplot(2, 2, 4)
plt.hist(array(w2.T.dot(x_blue))[0], color='b', bins=10, density=True)
plt.hist(array(w2.T.dot(x_red))[0], color='r', bins=10, density=True)

plt.show()

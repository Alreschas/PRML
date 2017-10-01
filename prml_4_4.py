#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 00:25:05 2017

@author: Narifumi
"""

import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random

# 訓練データ
N_blue = 50
N_red = 50
mu_red = [-1, 1]
mu_blue = [1, -1]
cov = [[1.2, 1], [1, 1.2]]
x_red, y_red = np.random.multivariate_normal(mu_red, cov, N_red).T
x_blue, y_blue = np.random.multivariate_normal(mu_blue, cov, N_blue).T

x1 = vstack((hstack((x_red, x_blue)).T, hstack((y_red, y_blue)).T)).T
t1 = np.matrix([[1, 0]] * N_red + [[0, 1]] * N_blue)

# 外れ値の追加
N_b_out = 10
mu_b_out = [7, -6]
cov2 = [[1, 0], [0, 1]]
x_b_out, y_b_out = np.random.multivariate_normal(mu_b_out, cov2, N_b_out).T
x_blue2, y_blue2 = np.r_[x_blue, x_b_out], np.r_[y_blue, y_b_out]

x2 = vstack((hstack((x_red, x_blue2)).T, hstack((y_red, y_blue2)).T)).T
t2 = np.matrix([[1, 0]] * N_red + [[0, 1]] * (N_blue + N_b_out))


def leastSquareMethod(x, t):
    X = np.ones([t.shape[0], 1])
    X = np.concatenate((X, x), axis=1)
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
    return W


def decisionBoundary(x, W):
    a = -(W[1, 1] - W[1, 0]) / (W[2, 1] - W[2, 0])
    b = -(W[0, 1] - W[0, 0]) / (W[2, 1] - W[2, 0])
    return a * x + b



# 外れ値なし
x_line = np.linspace(-4, 8, 1000)
plt.subplot(1, 2, 1)
plt.scatter(x_red, y_red, color='r', marker='x')
plt.scatter(x_blue, y_blue, color='none', marker='o', edgecolors='b')

W = leastSquareMethod(x1, t1)
plt.plot(x_line, decisionBoundary(x_line, W), 'g-')

xlim(-4.0, 9.0)
ylim(-9.0, 4.0)

# 外れ値あり
plt.subplot(1, 2, 2)
x_line = np.linspace(-4, 8, 1000)
plt.scatter(x_red, y_red, color='r', marker='x')
plt.scatter(x_blue2, y_blue2, color='none', marker='o', edgecolors='b')

W = leastSquareMethod(x2, t2)
plt.plot(x_line, decisionBoundary(x_line, W), 'g-')

xlim(-4.0, 9.0)
ylim(-9.0, 4.0)

plt.show()

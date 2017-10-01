#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 21:44:27 2017

@author: Narifumi
"""


import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random

# 訓練データ
N_red = 50
N_green = 50
N_blue = 50
mu_red = [-2, 2]
mu_green = [0, 0]
mu_blue = [2, -2]

cov = [[1.2, 1], [1, 1.2]]
x_red, y_red = np.random.multivariate_normal(mu_red, cov, N_red).T
x_green, y_green = np.random.multivariate_normal(mu_green, cov, N_green).T
x_blue, y_blue = np.random.multivariate_normal(mu_blue, cov, N_blue).T


x = vstack((hstack((x_red, x_green, x_blue)).T, hstack((y_red, y_green, y_blue)).T)).T
t = np.matrix([[0, 0, 1]] * N_red + [[0, 1, 0]] * N_green + [[1, 0, 0]] * N_blue)


def leastSquareMethod(x, t):
    X = np.ones([t.shape[0], 1])
    X = np.concatenate((X, x), axis=1)
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
    return W


def classification(x1, x2, W):
    x = np.matrix([[1], [x1], [x2]])
    c = W.T.dot(x)
    return np.argmax(c)


def decisionBoundary(x, W, c1, c2):
    a = -(W[1, c2] - W[1, c1]) / (W[2, c2] - W[2, c1])
    b = -(W[0, c2] - W[0, c1]) / (W[2, c2] - W[2, c1])
    return a * x + b


# 外れ値なし
x_line = np.linspace(-6, 6, 100)
y_line = np.linspace(-6, 6, 100)
X, Y = meshgrid(x_line, y_line)
plt.scatter(x_red, y_red, color='r', marker='x')
plt.scatter(x_green, y_green, color='g', marker='+')
plt.scatter(x_blue, y_blue, color='none', marker='o', edgecolors='b')

W = leastSquareMethod(x, t)
plt.plot(x_line, decisionBoundary(x_line, W, 0, 1), 'g-')
plt.plot(x_line, decisionBoundary(x_line, W, 1, 2), 'g-')
plt.plot(x_line, decisionBoundary(x_line, W, 0, 2), 'g-')

Z = np.zeros([100, 100])
for i in range(100):
    for j in range(100):
        Z[i, j] = classification(X[i, j], Y[i, j], W)
pcolor(X, Y, Z, alpha=0.2, edgecolor='white', lw=0)

xlim(-6.0, 6.0)
ylim(-6.0, 6.0)

plt.show()

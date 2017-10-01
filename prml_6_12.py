#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:26:37 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

data = np.loadtxt('data/classification.txt', delimiter=' ')
dataS = data.shape[0]

tdataS = 50
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, tdataS), np.linspace(-3, 3, tdataS))
Z = np.zeros(X.shape)


def pxc1(x):
    N = 2
    cov = np.array([[0.625, -0.2165], [-0.2165, 0.875]])
    mu = np.array([0, -0.1])
    invCov = np.linalg.inv(cov)
    denom = np.sqrt(np.linalg.det(cov)) * (2 * np.pi)**(N / 2)
    ret = np.exp(-0.5 * (x - mu).T.dot(invCov).dot(x - mu)) / denom
    return ret


def pxc2(x):
    N = 2
    mu1 = np.array([1, 1])
    mu2 = np.array([1, -1])

    cov1 = np.array([[0.2241, -0.1368], [-0.1368, 0.9759]])
    cov2 = np.array([[0.2375, 0.1516], [0.1516, 0.4125]])

    invCov1 = np.linalg.inv(cov1)
    invCov2 = np.linalg.inv(cov2)
    denom1 = np.sqrt(np.linalg.det(cov1)) * (2 * np.pi)**(N / 2)
    denom2 = np.sqrt(np.linalg.det(cov2)) * (2 * np.pi)**(N / 2)
    ret = 0.5 * np.exp(-0.5 * (x - mu1).T.dot(invCov1).dot(x - mu1)) / denom1
    ret += 0.5 * np.exp(-0.5 * (x - mu2).T.dot(invCov2).dot(x - mu2)) / denom2
    return ret


def plotdata():
    cmap = cm.get_cmap('coolwarm')
    norm = matplotlib.colors.Normalize()
    normed_data = norm(data[:, 2])
    rgba = cmap(normed_data)
    Z = np.zeros(X.shape)
    plt.scatter(data[:, 0], data[:, 1], marker='o', edgecolors=rgba, s=30, c='None', lw=1.0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = 1 / (1 + pxc2(x) / pxc1(x))
    plt.contour(X, Y, Z, [0.5], colors='g', linewidths=1.5)

######################


def kern(x1, x2):
    beta = 0.5
    ret = np.exp(-beta * (x1 - x2).dot(x1 - x2))
    return ret


def gramMat(x):
    size = x.shape[0]
    ret = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            ret[i, j] = kern(x[i, :].T, x[j, :].T)
    return ret


def sigmoid(a):
    ret = 1 / (1 + np.exp(-a))
    return ret


def kappa(var):
    ret = (1 + np.pi * var / 8)**(-1 / 2)
    return ret


nu = 0
WN = np.zeros([dataS, dataS])
aN = np.zeros(dataS)
sN = np.zeros(dataS)
k = np.zeros(dataS)
tN = data[:, 2]
K = gramMat(data[:, 0:2])
CN = K + np.eye(dataS) * nu

for epoch in range(100):
    for n in range(dataS):
        WN[n, n] = sigmoid(aN[n]) * (1 - sigmoid(aN[n]))
        sN[n] = sigmoid(aN[n])
    aN = CN.dot(np.linalg.inv(np.eye(dataS) + WN.dot(CN))).dot(tN - sN + WN.dot(aN))
H = WN + np.linalg.inv(CN)
print(aN)
print(CN.dot(tN - sN))

invWN = np.linalg.inv(WN)
inv_invWNpCN = np.linalg.inv(invWN + CN)

for i in range(tdataS):
    for j in range(tdataS):
        x = np.array([X[i, j], Y[i, j]])
        for n in range(dataS):
            k[n] = kern(data[n, 0:2], x)
        c = kern(x, x)
        mu = k.T.dot(tN - sN)
        var = c - k.T.dot(inv_invWNpCN).dot(k)
        Z[i, j] = sigmoid(kappa(var) * mu)

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, [0.5], colors='k', linewidths=1.5)
plotdata()
plt.xlim([-2.5, 2.5])
plt.ylim([-3, 3])
plt.subplot(1, 2, 2)
plt.pcolor(X, Y, Z, alpha=1.0, cmap=cm.coolwarm)
plt.contour(X, Y, Z, [0.5], colors='k', linewidths=1.5)
plt.xlim([-2.5, 2.5])
plt.ylim([-3, 3])

plt.show()

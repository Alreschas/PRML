#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


# 訓練データ
data = np.array(
    [[0.000000, 0.349486],
     [0.111111, 0.830839],
     [0.222222, 1.007332],
     [0.333333, 0.971507],
     [0.444444, 0.133066],
     [0.555556, 0.166823],
     [0.666667, -0.848307],
     [0.777778, -0.445686],
     [0.888889, -0.563567],
     [1.000000, 0.261502]])

x = data[:, 0]
t = data[:, 1]


# プロット用データ
plotS = 100
X = np.linspace(0, 1, plotS)
Y = np.zeros(plotS)


def _phi(xn, M):
    ret = np.zeros([M + 1])
    for m in range(M + 1):
        ret[m] += xn**m
    return ret


def _Phi(x, M):
    N = x.shape[0]
    ret = np.zeros([N, M + 1])
    for n in range(N):
        ret[n, :] = _phi(x[n], M)
    return ret


plotArea = 0
for M in [0, 1, 3, 9]:
    # wの学習
    Phi = _Phi(x, M)
    w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T).dot(t)

    plotArea += 1
    plt.subplot(2, 2, plotArea)

    # 訓練データのプロット
    plt.plot(x, t, 'o', c='w', ms=5, markeredgecolor='blue', markeredgewidth=1)

    # 真の曲線のプロット
    plt.plot(X, np.sin(2 * np.pi * X), 'g')

    # 近似曲線のプロット
    for i in range(plotS):
        Y[i] = w.dot(_phi(X[i], M))
    plt.plot(X, Y, 'r')

plt.show()

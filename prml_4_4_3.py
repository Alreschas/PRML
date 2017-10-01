#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:41:34 2017

@author: Narifumi
"""

# IRLSによるロジスティック回帰

import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random

plt.clf()


def gaussian(x, mu, sig):
    D = 2
    ret = np.exp(-0.5 * (x - mu).T.dot(np.linalg.inv(sig)).dot(x - mu))
    ret = ret / ((2 * np.pi)**(D / 2) * np.sqrt(np.linalg.det(sig)))
    return ret


def sigmoid(a):
    ret = 1 / (1 + exp(-a))
    return ret


def softmax(a, b, c):
    ret = np.exp(a) / (np.exp(a) + np.exp(b) + np.exp(c))
    return ret


def learn_ml(x, w, t, eta):
    N = x.shape[0]
    for n in range(N):
        xn = np.append([1], x[n])
        tn = t[n]
        yn = sigmoid(np.inner(w, xn))
        divE = (yn - tn) * xn
        w = w - eta * divE
    return w


def learn_irls(x, w, t):
    N = x.shape[0]
    x_bias = np.ones([N, 1])
    PHI = np.hstack([x_bias, x])
    R = np.zeros([N, N])
    y_v = np.zeros([N, 1])
    t_v = np.matrix(t).T
    for n in range(N):
        xn = PHI[n, :]
        y_v[n, 0] = sigmoid(np.inner(w, xn))
        R[n, n] = y_v[n, 0] * (1 - y_v[n, 0])
    grad = np.linalg.inv(PHI.T.dot(R).dot(PHI)).dot(PHI.T).dot(y_v - t_v)
    wnew = w
    wnew[0] -= grad[0, 0]
    wnew[1] -= grad[1, 0]
    wnew[2] -= grad[2, 0]
    return wnew


def makeData(N, bias):
    def h(x, y):
        return 1.0 * x - y + bias

    X = np.array([-0.5, bias - 0.5]) + np.random.random([N, 2])
    t = [1 if (0.2 * np.random.randn() + h(x, y)) < 0 else 0 for x, y in X]
    return X, t


def g(x):
    return 1.0 * x + bias


bias = 0.6
xmax = 0.6
xmin = -0.6
ymax = 0.6 + bias
ymin = -0.6 + bias

# 訓練データ
N = 200

# ガウス分布

# データをまとめる
X, t = makeData(N, bias)
N_red = sum(t)
N_blue = N - N_red

# プロット用データ
x_line = np.linspace(xmin, xmax, 100)
y_line = np.linspace(ymin, ymax, 100)
X_L, Y_L = meshgrid(x_line, y_line)


def f(x, w):
    ret = -(w[0] / w[2] + w[1] / w[2] * x)
    return ret


w1 = np.zeros(3)  # np.random.rand(3)
w2 = np.zeros(3)

# IRLSによる学習
for i in range(500):
    w1 = learn_irls(X, w1, t)
    eta = 0.1

# 勾配降下法による学習
for i in range(1000):
    w2 = learn_ml(X, w2, t, eta)
    eta *= 0.99
print(w1)
print(w2)

plt.subplot(1, 2, 1)
[plt.scatter(xn, yn, color='none', marker='o', edgecolors='r') if(tn == 1) else plt.scatter(xn, yn, color='none', marker='o', edgecolors='b') for tn, xn, yn in zip(t, X[:, 0], X[:, 1])]

plt.plot(x_line, f(x_line, w1), 'g')
# plt.plot(x_line,f(x_line,w2),'c')
plt.plot(x_line, g(x_line), 'r')

xlim(xmin, xmax)
ylim(ymin, ymax)

plt.subplot(1, 2, 2)

[plt.scatter(xn, yn, c=sigmoid(np.inner(w1, [1, xn, yn])), alpha=0.3, marker='o', vmin=0.0, vmax=1.0, s=30) for xn, yn in X]

plt.plot(x_line, f(x_line, w1), 'g')
# plt.plot(x_line,f(x_line,w2),'c')


xlim(xmin, xmax)
ylim(ymin, ymax)


# 事前分布
# pi1=N_red/(N)
# pi2=N_blue/(N)
# 平均
# mu1=(np.matrix(t).dot(X)/N_red).T
# mu2=((1-np.matrix(t)).dot(X)/N_blue).T
# 分散
# s1=0
# s2=0
# for i in range(N):
#    xtmp=np.matrix(X[i]).T
#    s1+=(xtmp-mu1).dot((xtmp-mu1).T)*t[i]/N_red
#    s2+=(xtmp-mu2).dot((xtmp-mu2).T)*(1-t[i])/N_blue
#s=N_red/N*s1 + N_blue/N*s2
#
# Z1=np.zeros(X_L.shape)
# Z2=np.zeros(X_L.shape)
# Z=np.zeros(X_L.shape)
# pc1=np.zeros(X_L.shape)
# pc2=np.zeros(X_L.shape)
#
# for i in range(X_L.shape[0]):
#    for j in range(X_L.shape[1]):
#        xtmp=np.matrix([[X_L[i,j]],[Y_L[i,j]]])
#        #sが共通の場合は、線形分離可能
#        Z1[i,j] = gaussian(xtmp,mu1,s)
#        Z2[i,j] = gaussian(xtmp,mu2,s)
#        a1=np.log((pi1*Z1[i,j])/(pi2*Z2[i,j]))
#        pc1[i,j] = sigmoid(a1)
#
# plt.contour(X_L,Y_L,Z1,alpha=0.3)
# plt.contour(X_L,Y_L,Z2,alpha=0.3)
# plt.contour(X_L,Y_L,pc1,alpha=0.3)
#

# Z=np.zeros(X.shape)
# for i in range(X.shape[0]):
#    for j in range(X.shape[1]):
#        x=np.matrix([1,X[i,j],Y[i,j]]).T
#        Z[i,j] = sigmoid(w2.T.dot(x))
# plt.pcolor(X,Y,Z)

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 06:13:32 2016

@author: Narifumi
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
a0 = -0.3
a1 = 0.5

plt.clf()


def h(x):
    ret = a0 + a1 * x
    return ret


def makeTarget(N):
    sig = 0.2
    mu = 0
    xt = 2 * np.random.rand(N) - 1.0
    yt = h(xt) + (mu + np.random.randn(N) * sig)
    return xt, yt


def basisFunc(x, m):
    # ダミーの基底関数
    if(m == 0):
        return np.ones(x.shape)
    ret = x
    return ret


def estFunc(x, w0, w1):
    ret = w0 * basisFunc(x, 0) + w1 * basisFunc(x, 1)
    return ret


def learn(xtNN, ytNN, mN, sN, beta):
    phi = np.zeros([2, 1])
    phi[0] = basisFunc(xtNN, 0)
    phi[1] = basisFunc(xtNN, 1)
    sNN = np.linalg.inv(np.linalg.inv(sN) + beta * phi.dot(phi.transpose()))
    mNN = sNN.dot(np.linalg.inv(sN).dot(mN) + beta * phi * ytNN)
    return mNN, sNN
#    return w


def gaussDist1D(x, mu, beta):
    ret = np.sqrt(beta / (2 * np.pi)) * np.exp(-beta / 2 * ((x - mu)**2))
    return ret


def gaussDist2D(x, y, mu, S):
    invS = np.linalg.inv(S)
    x1 = (x - mu[0])
    x2 = (y - mu[1])
    delta = x1 * (x1 * invS[0, 0] + x2 * invS[1, 0]) + x2 * (x1 * invS[0, 1] + x2 * invS[1, 1])
    ret = 1 / (2 * np.pi * np.sqrt(np.linalg.det(S))) * np.exp(-1 / 2 * delta)
    return ret


# 訓練データ数
N = 20

# 学習用パラメータ
M = 2  # 基底関数の数

# lnlam=-100
# プロット用データ
x = np.arange(-1.0, 1.0, 0.01)
y = h(x)

w0, w1 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))

alpha = 2
beta = 25

# 訓練データの生成
xt, yt = makeTarget(N)
# w=learn(xt,yt,M,alpha,beta)
# plt.plot(x,estFunc(x,w))
# plt.ylim([-1.5,1.5])

# 事前分布
sN = np.eye(2, 2) / alpha
mN = np.zeros([2, 1])
z = gaussDist2D(w0, w1, mN, sN)

plt.subplot(4, 3, 2)
plt.contourf(w0, w1, z, 100)
plt.subplot(4, 3, 3)
for i in range(6):
    w = np.random.multivariate_normal(mN.transpose().tolist()[0], sN)
    plt.plot(x, estFunc(x, w[0], w[1]), c='r')

# 学習1回目
mN, sN = learn(xt[0], yt[0], mN, sN, beta)
# z=z*gaussDist1D(yt[0],estFunc(xt[0],w0,w1),beta)
z = gaussDist2D(w0, w1, mN, sN)
lk = gaussDist1D(yt[0], estFunc(xt[0], w0, w1), beta)

# 最新データに対する尤度
plt.subplot(4, 3, 4)
#pylab.pcolor(w0, w1, lk, shading='flat')
plt.contourf(w0, w1, lk, 100)
plt.plot(a0, a1, 'w+', markeredgewidth=1)

# 事後分布
plt.subplot(4, 3, 5)
plt.contourf(w0, w1, z, 100)
plt.plot(a0, a1, 'w+', markeredgewidth=1)

# 多項式
plt.subplot(4, 3, 6)
for i in range(6):
    w = np.random.multivariate_normal(mN.transpose().tolist()[0], sN)
    plt.plot(x, estFunc(x, w[0], w[1]), c='r')
plt.plot(xt[0], yt[0], 'o', c='none', markeredgecolor='blue', markeredgewidth=1)


# 学習2回目
mN, sN = learn(xt[1], yt[1], mN, sN, beta)
# z=z*gaussDist1D(yt[1],estFunc(xt[1],w0,w1),beta)
z = gaussDist2D(w0, w1, mN, sN)
lk = gaussDist1D(yt[1], estFunc(xt[1], w0, w1), beta)

# 最新データに対する尤度
plt.subplot(4, 3, 7)
plt.contourf(w0, w1, lk, 100)
plt.plot(a0, a1, 'w+', markeredgewidth=1)

# 事後分布
plt.subplot(4, 3, 8)
plt.contourf(w0, w1, z, 100)
plt.plot(a0, a1, 'w+', markeredgewidth=1)

# 多項式
plt.subplot(4, 3, 9)
for i in range(6):
    w = np.random.multivariate_normal(mN.transpose().tolist()[0], sN)
    plt.plot(x, estFunc(x, w[0], w[1]), c='r')
plt.plot(xt[0], yt[0], 'o', c='none', markeredgecolor='blue', markeredgewidth=1)
plt.plot(xt[1], yt[1], 'o', c='none', markeredgecolor='blue', markeredgewidth=1)

# 学習N回目
for i in range(2, N):
    #    z=z*gaussDist1D(yt[i],estFunc(xt[i],w0,w1),beta)
    mN, sN = learn(xt[i], yt[i], mN, sN, beta)
lk = gaussDist1D(yt[i], estFunc(xt[i], w0, w1), beta)
z = gaussDist2D(w0, w1, mN, sN)

# 最新データに対する尤度
plt.subplot(4, 3, 10)
plt.contourf(w0, w1, lk, 100)
plt.plot(a0, a1, 'w+', markeredgewidth=1)

# 事後分布
plt.subplot(4, 3, 11)
plt.contourf(w0, w1, z, 100)
plt.plot(a0, a1, 'w+', markeredgewidth=1)

# 多項式
plt.subplot(4, 3, 12)
for i in range(6):
    w = np.random.multivariate_normal(mN.transpose().tolist()[0], sN)
    plt.plot(x, estFunc(x, w[0], w[1]), c='r')
for i in range(N):
    plt.plot(xt[i], yt[i], 'o', c='none', markeredgecolor='blue', markeredgewidth=1)

plt.show()

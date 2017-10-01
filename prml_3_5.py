#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 06:13:32 2016

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt

plt.clf()


def makeTarget(N):
    sig = 0.3
    mu = 0
#    xt=np.random.rand(N)
    xt = np.arange(0, 1 + 1 / N, 1 / (N - 1))
    yt = np.sin(xt * np.pi * 2) + (mu + np.random.randn(N) * sig)
    return xt, yt


def basisFunc(x, j, M):
    # ダミーの基底関数
    if(j == 0):
        return np.ones(x.shape)
    mu = (j - 1) / (M - 2)

#    mu=(j-1)/(M-1)
    sig2 = 1 / 250
    ret = np.exp(-(x - mu)**2 / (2 * sig2))
    return ret


def estFunc(x, w):
    M = w.shape[0]
    N = x.shape[0]
    phi = np.zeros([M, N])
    for m in range(M):
        phi[m] = basisFunc(x, m, M)
    ret = w.dot(phi)
    return ret


def learn(xt, yt, M, lnlam):
    lam = np.exp(lnlam)
    N = xt.shape[0]
    PHI = np.zeros([N, M])
    for n in range(N):
        for m in range(M):
            PHI[n, m] = basisFunc(xt[n], m, M)
#    w=((np.linalg.inv(np.transpose(PHI).dot(PHI))).dot(np.transpose(PHI))).dot(yt)

    w = ((np.linalg.inv(lam * np.eye(M, M) + np.transpose(PHI).dot(PHI))).dot(np.transpose(PHI))).dot(yt)
    return w


def h(x):
    ret = np.sin(x * 2 * np.pi)
    return ret


def calcBiasVariance(lnlam, L, N, M):
    bvN = N
    est = np.zeros([bvN, L])
    y_bar = np.zeros(bvN)
    for i in range(L):
        xt, yt = makeTarget(N)
        w = learn(xt, yt, M, lnlam)
        y_bar = y_bar + estFunc(xt, w) / L
        est[:, i] = estFunc(xt, w)

    bias = 0
    variance = 0
    bias += (y_bar - h(xt)).dot(y_bar - h(xt)) / bvN
    for i in range(L):
        variance += ((y_bar - est[:, i]).dot(y_bar - est[:, i])) / (bvN * L)

    return bias, variance


# 訓練データ数
N = 25

# 学習用パラメータ
M = 25  # 基底関数の数
L = 1000  # データセットの数
lnlam = -2.4
# lnlam=-100
# プロット用データ
x = np.arange(-0.2, 1.2, 0.01)
y = h(x)


# 訓練データの生成
mean = np.zeros(x.shape[0])
for i in range(L):
    xt, yt = makeTarget(N)
    w = learn(xt, yt, M, lnlam)
    if(i < 20):
        #        plt.plot(xt,yt,'o')
        plt.plot(x, estFunc(x, w), 'gray', alpha=0.3)
    mean = mean + estFunc(x, w) / L

plt.plot(x, y)
plt.plot(x, mean, 'r')
plt.ylim([-1.5, 1.5])

# 基底関数
# for j in range(M):
#    plt.plot(x,basisFunc(x,j,M))

# バイアスーバリアンス
# L=100
#bias = np.zeros(N)
# variance=np.zeros(N)
# lamMin=-3
# lamMax=2
#
#lnlam = np.arange(lamMin,lamMax,(lamMax-lamMin)/N)
# for i in range(N):
#    bias[i],variance[i] = calcBiasVariance(lnlam[i],L,N,M)
#
# plt.plot(lnlam,bias)
# plt.plot(lnlam,variance)
# plt.plot(lnlam,bias+variance)

plt.show()

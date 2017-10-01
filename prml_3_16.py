#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:26:36 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt

# 訓練データ
D = np.loadtxt("data/curvefitting.txt")
x = D[:, 0]
t = D[:, 1]

#x = np.random.rand(100)
#t=np.sin(x*2*np.pi)+np.sqrt(1/11.1) * np.random.randn(x.shape[0])


M = 10  # k基底関数の数(バイアスを含む)
N = x.shape[0]


def basisFunc(x, i):
    mu = 0
    sig2 = (1 / M)**2
    if i == 0:
        return 1
    if M == 2:
        mu = 0.5
    else:
        mu = (i - 1) / (M - 2)
    ret = np.exp(-(x - mu)**2 / (2 * sig2))
    return ret


def phi(x):
    ret = np.zeros([M, 1])
    for i in range(M):
        ret[i] = basisFunc(x, i)
    return ret


def PHI(x):
    ret = np.zeros([N, M])
    for i in range(N):
        ret[i, :] = phi(x[i]).transpose()
    return ret


def MAP(x, t, alpha, beta):
    sn = np.linalg.inv(alpha * np.eye(M) + beta * PHI(x).transpose().dot(PHI(x)))
    mn = beta * sn.dot(PHI(x).transpose()).dot(t)
    return mn, sn


def f(x, w):
    ret = w.transpose().dot(phi(x))
    return ret


def gamma(x, alpha, beta):
    PTP = PHI(x).transpose().dot(PHI(x))
    # 固有値、固有ベクトルを計算
    lam, v = np.linalg.eig(PTP)
    gam = 0
    for lami in lam:
        gam += beta * lami / (alpha + beta * lami)
    return gam


def alphaEw(w, alpha):
    ret = alpha * w.dot(w)
    return ret


# プロット用データ
gN = 100
gx = np.linspace(-0.0, 1.0, gN)
gt = np.sin(2 * np.pi * gx)

beta = 11.1

# 曲線プロット
#alpha = 5.0e-1
# mn,sn=MAP(x,t,alpha,beta)
#gy = np.zeros(gN)
# for i in range(gN):
#    gy[i] = f(gx[i],mn)
#
# plt.plot(x,t,'o')
# plt.plot(gx,gt)
# plt.plot(gx,gy)
# plt.ylim([-1.5,1.5])

# γ-2αEwプロット
#lnalpha = np.linspace(-5,5,gN)
#gam = np.zeros(gN)
#alphaew = np.zeros(gN)
#
# for i in range(gN):
#    alpha = np.exp(lnalpha[i])
#    gam[i] = gamma(x,alpha,beta)
#    mn,sn = MAP(x,t,alpha,beta)
#    alphaew[i] = alphaEw(mn,alpha)
#
# plt.plot(lnalpha,gam)
# plt.plot(lnalpha,alphaew)
# plt.ylim([0.0,25])

# wi-γプロット
alpha = np.exp(np.linspace(-5, 15, gN))
gam = np.zeros(gN)
w = np.zeros([M, gN])
for i in range(gN):
    gam[i] = gamma(x, alpha[i], beta)
    w[:, i], sn = MAP(x, t, alpha[i], beta)

for i in range(M):
    plt.plot(gam, w[i, :])


# 基底関数プロット
#by = np.zeros([M,gN])
# for j in range(gN):
#    by[:,j] = phi(gx[j])[:,0]
#
# for i in range(M):
#    plt.plot(gx,by[i,:])

plt.show()

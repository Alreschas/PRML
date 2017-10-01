#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:46:36 2016

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sig2):
    p = (1 / np.sqrt(2 * np.pi * sig2)) * np.exp(-(x - mu)**2 / (2 * sig2))
    return p


beta = 11.1
alpha = 5.0e-3

step = 0.1
sig2 = 0.1
data = np.loadtxt('data/curvefitting.txt', comments='#', delimiter=' ')
x_t = data[:, 0]
y_t = data[:, 1]

# x_t=np.arange(0,1+step,step)
# y_t=np.sin(2*np.pi*x_t)+np.random.randn(x_t.shape[0])*np.sqrt(sig2)

N = x_t.shape[0]
plt.plot(x_t, y_t, 'o', ms=10, alpha=0.8, c='w', markeredgecolor='blue', markeredgewidth=2)

# 多項式の次数
M = 9


def phi(x):
    ret = np.zeros([M + 1, x.shape[0]])
    for i in range(M + 1):
        ret[i, :] = x**i
    return ret


def expDist(x, x_t, y_t):
    phix = phi(x)
    phixt2 = np.zeros([M + 1, M + 1])
    phixtxtn = np.zeros([M + 1, 1])
    sig2 = np.zeros([x.shape[0], 1])
    for i in range(N):
        phixtxtn += phi(x_t)[:, i:i + 1] * y_t[i]
        phixt2 += phi(x_t)[:, i:i + 1].dot(phi(x_t)[:, i:i + 1].T)
    Sinv = alpha * np.identity(M + 1) + beta * phixt2
    S = np.linalg.inv(Sinv)
    mu = beta * ((phix.T).dot(S)).dot(phixtxtn)
    for i in range(x.shape[0]):
        sig2[i, 0] = 1 / beta + (phix[:, i:i + 1].T).dot(S).dot(phix[:, i:i + 1])
    return mu, sig2


x = np.arange(-1, 2, 0.01)
expDist(x, x_t, y_t)
m, s2 = expDist(x, x_t, y_t)
plt.plot(x, m, 'r')

plt.plot(x, np.sin(2 * np.pi * x), 'g')
# plt.plot(x,np.sin(2*np.pi*x)+np.sqrt(sig2),'r',alpha=0.3)
# plt.plot(x,np.sin(2*np.pi*x)-np.sqrt(sig2),'r',alpha=0.3)


plt.ylim(-1.5, 1.5)

xs = np.concatenate((x, x[::-1]))
ys = np.concatenate((m + np.sqrt(s2), (m - np.sqrt(s2))[::-1]))
p = plt.fill(xs, ys, facecolor='pink', alpha=0.5, lw=0)

plt.show()

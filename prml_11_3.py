#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:54:08 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt


def dist_exp(y, lam):
    py = lam * np.exp(-lam * y)
    return py


def rand_exp(N, lam):
    z = np.random.rand(N)
    y = -np.log(1 - z) / lam
    return y


def dist_cauchy(y):
    py = 1 / (np.pi * (1 + y**2))
    return py


def rand_cauchy(N):
    z = np.random.rand(N)
    y = np.tan(np.pi * (z - 0.5))
    return y


def dist_gauss(y):
    py = np.exp(-0.5 * (y**2)) / np.sqrt(2 * np.pi)
    return py


def rand_gauss(N):
    z = np.random.rand(2, N) * 2 - 1
    for i in range(N):
        while(z[0, i]**2 + z[1, i]**2 > 1):
            z[:, i] = np.random.rand(2) * 2 - 1
    r2 = z[0, :]**2 + z[1, :]**2
    y = np.zeros([2, N])
    y[0, :] = z[0, :] * (-2 * np.log(r2) / r2)**0.5
    y[1, :] = z[1, :] * (-2 * np.log(r2) / r2)**0.5
    return y


N = 100000
xMax = 5
bins = 200

plt.subplot(2, 2, 1)
plt.xlim([0, xMax])
x = np.linspace(0, xMax, 1000)
lam = 1
plt.plot(x, dist_exp(x, lam), 'r')
plt.hist(rand_exp(N, lam), bins=bins, range=[0, 5 * xMax], density=True, alpha=0.2)

plt.subplot(2, 2, 2)
plt.xlim([-xMax, xMax])
x = np.linspace(-xMax, xMax, 1000)
plt.plot(x, dist_cauchy(x), 'r')
plt.hist(rand_cauchy(N), bins=bins, range=[-5 * xMax, 5 * xMax], density=True, alpha=0.2)

plt.subplot(2, 2, 3)
plt.xlim([-xMax, xMax])
x = np.linspace(-xMax, xMax, 1000)
y = rand_gauss(N)
plt.plot(x, dist_gauss(x), 'r')
plt.hist(y[0, :], bins=bins, range=[-5 * xMax, 5 * xMax], density=True, alpha=0.2)

# plt.hist2d(y[0,:],y[1,:],bins=50)

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:21:01 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt


def distf_cauchy(z, b, c):
    pz = 1 / (np.pi * b * (1 + ((z - c) / b)**2))
    return pz


def distf_gam(z, a, b):
    pz = ((b**a) * (z**(a - 1)) * np.exp(-b * z)) / np.math.gamma(a)
    return pz


def rand_cauchy(N, b, c):
    z = (np.random.rand(N) - 0.5) * np.pi
    y = b * np.tan(z) + c
#    y = np.random.randn(N)/np.random.randn(N) * b + c
    return y


def rand_gam(N, a_gam, b_gam, b_cauchy, c_cauchy, k):

    z0 = rand_cauchy(N, b_cauchy, c_cauchy)
    y = np.random.rand(N) * (k * distf_cauchy(z0, b_cauchy, c_cauchy))
    for i in range(N):
        while(z0[i] < 0 or y[i] > distf_gam(z0[i], a_gam, b_gam)):
            z0[i] = rand_cauchy(1, b_cauchy, c_cauchy)
            y[i] = np.random.rand(1) * (k * distf_cauchy(z0[i], b_cauchy, c_cauchy))

    return z0


bins = 50
N = 10000
zMin = 0
zMax = 30
z = np.linspace(zMin, zMax, 500)

a_gam = 10
b_gam = 1

b_cauchy = np.sqrt(2 * a_gam - 1) / b_gam
c_cauchy = (a_gam - 1) / b_gam

k = np.pi * b_cauchy * b_gam * ((a_gam - 1)**(a_gam - 1)) * np.exp(-(a_gam - 1)) / np.math.gamma(a_gam)

plt.subplot(1, 2, 1)
plt.plot(z, k * distf_cauchy(z, b_cauchy, c_cauchy), color='green')
plt.plot(z, distf_gam(z, a_gam, b_gam), color='magenta')

plt.subplot(2, 2, 2)
plt.xlim([zMin, zMax])
plt.plot(z, distf_cauchy(z, b_cauchy, c_cauchy), color='green')
plt.hist(rand_cauchy(N, b_cauchy, c_cauchy), range=(-50, 50), bins=2 * bins, normed=True, alpha=0.3, color='green')

plt.subplot(2, 2, 4)
plt.xlim([zMin, zMax])
plt.plot(z, distf_gam(z, a_gam, b_gam), color='magenta')
plt.hist(rand_gam(N, a_gam, b_gam, b_cauchy, c_cauchy, k), range=(0, 50), bins=bins, normed=True, alpha=0.3, color='magenta')

plt.show()

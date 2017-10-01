#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 04:26:11 2016

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def gauss_dist(x, mu, lam):
    ret = np.sqrt(lam / (2 * np.pi)) * np.exp(-lam * (x - mu)**2 / 2)
    return ret


def t_dist(x, mu, lam, v):
    ret = (gamma((1 + v) / 2) / gamma(v / 2)) * (lam / (np.pi * v))**0.5 * (1 + lam * (x - mu)**2 / v)**(-(1 + v) / 2)
    return ret


x = np.arange(-10, 10, 0.1)


mu = 0
lam = 1
v = 1
plt.plot(x, t_dist(x, mu, lam, v))

mu = 0
lam = 1
v = 0.1
plt.plot(x, t_dist(x, mu, lam, v))

mu = 0
lam = 1
v = 100
plt.plot(x, t_dist(x, mu, lam, v))
plt.plot(x, gauss_dist(x, mu, lam))

plt.show()

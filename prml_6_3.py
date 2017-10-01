#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 23:15:36 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt


def realFunc(x):
    ret = np.sin(x * 2 * np.pi)
    return ret


def gauss(x1, x2, sig):
    ret = np.exp(-(1 / (2 * sig**2)) * np.linalg.norm(x1 - x2)**2)
    return ret


def normGauss(x, no, xt, sig):
    dataS = xt.shape[0]
    tot = 0
    ret = 0
    for i in range(dataS):
        tot += gauss(x, xt[i], sig)
    ret = gauss(x, xt[no], sig) / tot
    return ret


def nadarayaWatson(x, xt, yt, sig):
    dataS = xt.shape[0]
    mu = 0
    var = sig**2
    for no in range(dataS):
        mu += yt[no] * normGauss(x, no, xt, sig)
        var += (yt[no]**2) * normGauss(x, no, xt, sig)
    var -= (mu**2)
    return mu, var


sig = 0.05
dataS = 10
xt = np.linspace(0, 1, dataS)
yt = realFunc(xt) + np.random.randn(dataS) * 0.3
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(xt, yt, 'bo')
for i in range(dataS):
    circle = plt.Circle((xt[i], yt[i]), sig, fc="None")
    ax.add_patch(circle)

tdS = 100
x = np.linspace(-0.1, 1.1, tdS)
y = np.zeros(tdS)
var = np.zeros(tdS)
for i in range(tdS):
    y[i], var[i] = nadarayaWatson(x[i], xt, yt, sig)
plt.plot(x, y, 'r')
plt.fill_between(x, y + 2 * np.sqrt(var), y - 2 * np.sqrt(var), facecolor='pink', alpha=0.5, lw=0)

plt.plot(x, realFunc(x), 'g')

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:50:18 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt

plt.clf()

xmin = -10
xmax = 10
num = 100
x = np.linspace(xmin, xmax, num)


def probit(x):
    ret = 0.0
    div = 100
    for i in range(div):
        x2 = (x * i / (div - 1))**2
        ret += np.exp(-1 / 2 * x2)
    ret = 0.5 + ret / (np.sqrt(2 * np.pi)) * x / div
    return ret


def logisticSigmoid(x):
    ret = 1 / (1 + np.exp(-x))
    return ret


lam = np.sqrt(np.pi / 8)
plt.plot(x, logisticSigmoid(x), 'r-')
plt.plot(x, probit(lam * x), 'b--')
plt.xlim([xmin, xmax])
plt.ylim([0, 1])
plt.plot([xmin, xmax], [0.5, 0.5], 'k--')
plt.plot([0, 0], [0, 1], 'k--')

plt.show()

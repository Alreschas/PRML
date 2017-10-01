#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:47:57 2016

@author: Narifumi
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def von_mises(theta, theta0, m):
    dx = 0.001
    x = np.arange(0, 2 * np.pi, dx)
    I0 = (np.exp(m * np.cos(x)) * dx).sum() / (np.pi * 2)
    ret = 1 / (I0 * 2 * np.pi) * np.exp(m * np.cos(theta - theta0))
    return ret


theta0 = np.pi / 2
theta = np.arange(0, 2 * np.pi, 0.01)
m = 1
x = np.cos(theta)
y = np.sin(theta)
z = von_mises(theta, theta0, m)

plt.plot(theta, von_mises(theta, theta0, m))
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x, y, z)

plt.show()

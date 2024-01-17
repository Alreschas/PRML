#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:05:14 2016

@author: Narifumi
"""

# 二項分布と中心極限定理

import numpy as np
import matplotlib.pyplot as plt

N = 2
sample = 10000


plt.subplot(2, 1, 1)
a = np.random.binomial(N, 0.5, sample) / N
plt.hist(a, bins=20, density=True)
plt.xlim(0, 1.0)

plt.subplot(2, 1, 2)
a = np.random.rand(N, sample) / N
b = np.sum(a, axis=0)
plt.hist(b, bins=20, density=True)
plt.xlim(0, 1.0)

plt.show()

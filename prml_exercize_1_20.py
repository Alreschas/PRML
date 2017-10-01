#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:30:31 2016

@author: Narifumi
"""

from scipy.special import gamma
from matplotlib import pyplot as plt
import numpy as np

C = 500
r = np.linspace( 0.01, 7.0, C)

def density(D,sig2,r):
    sd=2*np.pi**(D/2.)/gamma(D/2.)
    pr=sd/((2*np.pi*sig2)**(D/2.))*np.exp(-r**2/(2*sig2))*r**(D-1)
    return pr

plt.plot(x, density(1,1,r))
plt.plot(x, density(2,1,r))
plt.plot(x, density(20,1,r))
plt.show()
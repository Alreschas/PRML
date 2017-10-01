#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:35:12 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt


def sig(z):
    ret = 1/(1 + np.exp(-z))
    return ret

def real_dist(z):
    ret = np.exp(-(z**2)/2)*sig(20*z+4)
    return ret

def d1_real_dist(z):
    ret = -z*real_dist(z)+20*real_dist(z)*(1-sig(20*z+4))
    return ret

def d2_real_dist(z):
    s = sig(20*z+4)
    ret = (20*(1-s) - z) * d1_real_dist(z) - (1+400*(s*(1-s)))*real_dist(z)
    return ret

def gauss(z,z0,beta):
    ret = np.sqrt(beta/(2*np.pi)) * np.exp(-(beta/2) * (z - z0)**2)
    return ret

#勾配法でモードz0を見つける
z0 = 2;
for i in range(100):
    z0 = z0 + 0.5 * d1_real_dist(z0)

A = -d2_real_dist(z0)/real_dist(z0)

num=1000
zmin = -2
zmax = 4
z = np.linspace(zmin, zmax, num)
Z = real_dist(z).sum() * ((zmax-zmin)/num)

#plt.subplot(1,2,1)
plt.plot(z,real_dist(z)/Z)
plt.plot(z,gauss(z,z0,A))
#plt.subplot(1,2,1)
#plt.plot(z,d1_real_dist(z))
#plt.subplot(1,2,2)
#plt.plot(z,d2_real_dist(z))



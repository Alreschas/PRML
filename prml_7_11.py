#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:23:01 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt


def multi_gauss(x, sig):
    ret = x.dot(np.linalg.inv(sig)).dot(x)
    return ret


def basisFunc1(x):
    mu = 2.
    beta = 1.
    ret = np.exp(-beta * (x - mu).dot(x - mu))
    return ret


def basisFunc2(x):
    mu = 3
    beta = 1.
    ret = np.exp(-beta * (x - mu).dot(x - mu))
    return ret


def lam(alpha1, alpha2, phi1, phi2, beta, t):
    invCm1 = np.linalg.inv(np.eye(phi2.shape[0]) / beta + np.outer(phi2, phi2) / alpha2)
    q1 = phi1.dot(invCm1).dot(t)
    s1 = phi1.dot(invCm1).dot(phi1)
#    print(q1**2,s1)
    ret = 0.5 * (np.log(alpha1)
                 - np.log(alpha1 + s1)
                 + (q1**2) / (alpha1 + s1))
    return ret


# 訓練データ
t_real = np.array([-1.0, 1.0])
x1 = np.array([1])
x2 = np.array([3])

# 中心wTφとしたときの、tの分散の逆数
beta = 1 / 0.1
# wの事前分布の分散の逆数
#alpha1 = 1/10
alpha2 = 1 / 4


dataS = 100
t1 = np.linspace(-2, 2, dataS)
t2 = np.linspace(-2, 2, dataS)
T1, T2 = np.meshgrid(t1, t2)
Z1 = np.zeros([dataS, dataS])
Z2 = np.zeros([dataS, dataS])


phi1 = np.array([basisFunc1(x1), basisFunc1(x2)])
phi2 = np.array([basisFunc2(x1), basisFunc2(x2)])

x = np.linspace(-1, 5, dataS)

alphaS = 1000
plt.subplot(1, 2, 2)
# plt.ylim([-5,1])
# plt.xlim([-5,5])
lna = np.linspace(-15, 10, alphaS)
lamalpha = np.zeros([alphaS])
for i in np.arange(alphaS):
    alpha = np.exp(lna[i])
    lamalpha[i] = lam(alpha, alpha2, phi1, phi2, beta, t_real)
plt.plot(lna, lamalpha)

invCm1 = np.linalg.inv(np.eye(phi2.shape[0]) / beta + np.outer(phi2, phi2) / alpha2)
q1 = phi1.dot(invCm1).dot(t_real)
s1 = phi1.dot(invCm1).dot(phi1)

alpha1 = 100000
if(q1**2 > s1):
    alpha1 = (s1**2) / (q1**2 - s1)
    print(np.log(alpha1))
    plt.axvline(x=np.log(alpha1), c='r')

ypp = np.zeros([dataS])
ymm = np.zeros([dataS])
ypm = np.zeros([dataS])
ymp = np.zeros([dataS])

for i in range(dataS):
    ypp[i] = + np.sqrt(1 / alpha1) * basisFunc1(np.array([x[i]]))\
             + np.sqrt(1 / alpha2) * basisFunc2(np.array([x[i]]))
    ymm[i] = - np.sqrt(1 / alpha1) * basisFunc1(np.array([x[i]]))\
             - np.sqrt(1 / alpha2) * basisFunc2(np.array([x[i]]))
    ypm[i] = + np.sqrt(1 / alpha1) * basisFunc1(np.array([x[i]]))\
        - np.sqrt(1 / alpha2) * basisFunc2(np.array([x[i]]))
    ymp[i] = - np.sqrt(1 / alpha1) * basisFunc1(np.array([x[i]]))\
        + np.sqrt(1 / alpha2) * basisFunc2(np.array([x[i]]))

plt.subplot(2, 2, 1)
plt.ylim([-4, 4])

# wが標準偏差のときの基底関数
plt.plot(x, ypp)
plt.plot(x, ymm)
plt.plot(x, ymp)
plt.plot(x, ypm)

plt.fill_between(x, ymm - np.sqrt(1 / beta), ypp + np.sqrt(1 / beta), alpha=0.5, color='pink')


plt.plot(x1, t_real[0], 'o')
plt.plot(x2, t_real[1], 'o')
plt.plot(x1, 0, 'x')
plt.plot(x2, 0, 'x')

sig1 = + np.outer(phi1, phi1) / alpha1\
       + np.outer(phi2, phi2) / alpha2\
       + np.eye(2) / beta
sig2 = + np.outer(phi2, phi2) / alpha2\
       + np.eye(2) / beta

# 共分散行列の行列式が1になるように調整
norm = 1 / np.sqrt(np.linalg.det(sig1))
sig1 = sig1 * norm
sig2 = sig2 * norm

for i in range(dataS):
    for j in range(dataS):
        t = np.array([T1[i, j], T2[i, j]])
        Z1[i, j] = multi_gauss(t, sig1)
        Z2[i, j] = multi_gauss(t, sig2)
plt.subplot(2, 2, 3)
plt.axis('equal')
plt.ylim([-2, 2])
plt.xlim([-1.5, 1.5])

plt.contour(T1, T2, Z1, [1, 5], colors="r")
plt.contour(T1, T2, Z2, [1, 5], colors='g')
plt.quiver(0, 0, phi1[0], phi1[1], angles='xy', scale_units='xy', scale=1)
plt.text(phi1[0], phi1[1], r'$\phi_1$')

plt.quiver(0, 0, phi2[0], phi2[1], angles='xy', scale_units='xy', scale=1)
plt.text(phi2[0], phi2[1], r'$\phi_2$')

plt.plot(t_real[0], t_real[1], 'x')

plt.show()

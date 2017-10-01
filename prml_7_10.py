#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:03:20 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt

useNormC = True


def multi_gauss(x, sig):
    global useNormC
    if(useNormC):
        ret = x.dot(np.linalg.inv(sig)).dot(x)
    else:
        D = sig.shape[0]
        norm = 1 / (((2 * np.pi)**(0.5 * D)) * (np.linalg.det(sig) ** 0.5))
        ret = np.exp(-0.5 * x.dot(np.linalg.inv(sig)).dot(x)) * norm
    return ret


def basisFunc1(x):
    mu = 1
    beta = 1
    ret = np.exp(-beta * (x - mu).dot(x - mu))
    return ret


def basisFunc2(x):
    mu = 2
    beta = 1
    ret = np.exp(-beta * (x - mu).dot(x - mu))
    return ret


# 訓練データ
t_real = np.array([-1.0, 1.0])
x1 = np.array([1])
x2 = np.array([3])

# 中心wTφとしたときの、tの分散の逆数
beta = 1 / 0.1
# wの事前分布の分散の逆数
alpha1 = 1 / 0.001
alpha2 = 1 / 1


dataS = 100
t1 = np.linspace(-2, 2, dataS)
t2 = np.linspace(-2, 2, dataS)
T1, T2 = np.meshgrid(t1, t2)
Z1 = np.zeros([dataS, dataS])
Z2 = np.zeros([dataS, dataS])


phi1 = np.array([basisFunc1(x1), basisFunc1(x2)])
phi2 = np.array([basisFunc2(x1), basisFunc2(x2)])


x = np.linspace(-1, 5, dataS)
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


plt.subplot(1, 2, 1)
plt.ylim([-1.5, 1.5])

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
sig2 = np.eye(2) / beta

# 共分散行列の行列式が1になるように調整
if(useNormC):
    norm = 1 / np.sqrt(np.linalg.det(sig1))
    sig1 = sig1 * norm
#    sig2 = sig2 * norm
    print(np.linalg.det(sig1))


for i in range(dataS):
    for j in range(dataS):
        t = np.array([T1[i, j], T2[i, j]])
        Z1[i, j] = multi_gauss(t, sig1)
        Z2[i, j] = multi_gauss(t, sig2)
plt.subplot(1, 2, 2)
plt.axis('equal')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])

if(useNormC):
    plt.contour(T1, T2, Z1, [1], colors="r")
    plt.contour(T1, T2, Z2, [1], colors='g')
else:
    plt.contour(T1, T2, Z1, [0.05], colors="r")
    plt.contour(T1, T2, Z2, [0.05], colors='g')
plt.quiver(0, 0, phi1[0], phi1[1], angles='xy', scale_units='xy', scale=1)
plt.text(phi1[0], phi1[1], r'$\phi_1$')

plt.quiver(0, 0, phi2[0], phi2[1], angles='xy', scale_units='xy', scale=1)
plt.text(phi2[0], phi2[1], r'$\phi_2$')

plt.plot(t_real[0], t_real[1], 'x')

plt.show()

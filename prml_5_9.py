#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:52:30 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from scipy import optimize

plt.clf()


def aFunc_l1(x):
    ret = np.tanh(x)
    return ret


def aFunc_l2(x):
    ret = x
    return ret


def out_l1(x, w_l1):
    ret = np.matrix(np.append(1, aFunc_l1(w_l1.T.dot(x)))).T
    return ret


def out_l2(x, w_l2):
    ret = np.matrix(aFunc_l2(w_l2.T.dot(x))).T
    return ret


def J(param, *args):
    """最小化を目指すコスト関数を返す"""
    w = param
    wl1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
    wl2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)
    # パラメータ以外のデータはargsを通して渡す
    xt, yt = args
    N = xt.shape[0]
    E = 0
    for n in range(N):
        z_l0 = np.matrix(np.append(1, xt[n])).T
        z_l1 = out_l1(z_l0, wl1)
        z_l2 = out_l2(z_l1, wl2)
        y = z_l2[0, 0]
        E += (y - yt[n])**2
    return E / 2


def gradient(param, *args):
    """コスト関数の偏微分を返す
    各パラメータで偏微分した関数リストを返す"""
    w = param
    wl1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
    wl2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)

    dE1 = np.zeros(wl1.shape)
    dE2 = np.zeros(wl2.shape)

    xt, yt = args
    N = xt.shape[0]

    for n in range(N):
        #        #forward prop
        z_l0 = np.matrix(np.append(1, xt[n])).T
        z_l1 = out_l1(z_l0, wl1)
        z_l2 = out_l2(z_l1, wl2)
        y = z_l2

        delta_l2 = np.matrix(y - yt[n]).T
        delta_l1 = np.multiply(1 - np.power(z_l1, 2), wl2.dot(delta_l2))

        dE_l2 = z_l1.dot(delta_l2.T)
        dE_l1 = z_l0.dot(delta_l1[1:, 0].T)

        dE2 += dE_l2
        dE1 += dE_l1
    dE = np.append(dE1.reshape([(N_in + 1) * N_hdn, 1]), dE2.reshape([(N_hdn + 1) * N_out, 1]))
    return dE


# 訓練データ
data = np.loadtxt('data/curvefitting.txt', comments='#', delimiter=' ')
x_t = data[:, 0]
y_t = data[:, 1]
N = 10

# 学習パラメータ
N_out = 1
N_in = 1
N_hdn = 1
grpNo = 1
for N_hdn in (1, 3, 10, 50):
    print("START:", N_hdn)

    w = np.random.randn((N_in + 1) * N_hdn + (N_hdn + 1) * N_out) * np.sqrt(10)
    # 確率的勾配降下法
#    Err_bk=0
#    Err=J(w,*(x_t,y_t))
#    i=0
#    while(np.fabs(Err-Err_bk)>1e-7 and i < 50000):
#        w = w - 0.1*gradient(w,*(x_t,y_t))/(N_hdn)
#        Err_bk=Err
#        Err=J(w,*(x_t,y_t))
#        if(i%500==0):print(i,Err)
#        i+=1
    # 共役勾配法
    w = optimize.fmin_cg(J, w, fprime=gradient, args=(x_t, y_t))
#w = optimize.fmin_bfgs(J, w, fprime=gradient, args=(x_t,y_t))
#w = optimize.fmin_cg(J, w, args=args,gtol=0)

    w_l1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
    w_l2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)

    # プロット
    xd = np.arange(-0., 1.001, 0.001)
    yd = np.zeros(xd.shape)
    zd = np.zeros([xd.shape[0], N_hdn + 1])

    plt.subplot(2, 2, grpNo)
    grpNo += 1
    plt.plot(x_t, y_t, 'o')
    for i in range(xd.shape[0]):
        z_l0 = np.matrix(np.append(1, xd[i])).T
        z_l1 = out_l1(z_l0, w_l1)
        z_l2 = out_l2(z_l1, w_l2)
        yd[i] = z_l2[0, 0]
        zd[i] = z_l1.T
    plt.plot(xd, yd)
    for i in range(N_hdn + 1):
        plt.plot(xd, zd[:, i], '--')
    plt.xlim([-0.1, 1.1])

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#プロット用データ
plotS = 1000
X = np.linspace(-1,1,plotS)

#ガウス分布の確率密度関数
def gaussDist(x,mu,sig2):
    ret = np.exp(-(x-mu)**2/(2*sig2))/np.sqrt(2*np.pi*sig2)
    return ret

# 真の分布
mu_r = 0
sig2_r = 0.05
Y_r = gaussDist(X,mu_r,sig2_r)

np.random.seed(10)
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.ylim([0,7])

    #訓練データ
    N = 2
    x_t = np.random.randn(N) * np.sqrt(sig2_r) + mu_r
    plt.plot(x_t,np.zeros(x_t.shape[0]),'bo')

    #最尤推定した分布
    mu_ML = x_t.sum()/N
    sig2_ML = ((x_t - mu_ML)**2).sum()/N
    Y_ml = gaussDist(X,mu_ML,sig2_ML)
    plt.plot(X,Y_ml,'r')
    
    #真の分布
    plt.plot(X,Y_r,'g')
#plt.savefig("/Users/Narifumi/Desktop/test.png")
#plt.figure("/Users/Narifumi/Desktop/test.png")

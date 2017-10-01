#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 16:02:28 2016

@author: Narifumi
"""
import numpy as np
import matplotlib.pyplot as plt


def gauss(x,mu,sig2):
    ret = 1/np.sqrt(2*np.pi*sig2) * np.exp(-(x-mu)**2/(2 * sig2))
    return ret
    
def bayse(xt,sig2_r,mu_pr,sig2_pr):
    TN = np.size(xt)
    mu_ml = xt.sum()/TN
    mu_N = sig2_r * mu_pr/(TN*sig2_pr + sig2_r) + mu_ml * TN* sig2_pr/(TN*sig2_pr+sig2_r)
    sig2_N= 1/(1/sig2_pr + TN/sig2_r)
    return mu_N,sig2_N

#トレーニングデータ数
TN=10

#xの分布
#p(x|mu_r,sig2_r)
mu_r = 0.8
sig2_r = 0.1

#muの事前分布
#p(mu|mu_pr,sig2_pr)
mu_pr = 0
sig2_pr = 0.1

#トレーニングデータ
xt=mu_r + np.random.randn(TN) * np.sqrt(sig2_r)

x = np.arange(-1,1,0.01)
plt.plot(x,gauss(x,0,0.1))

plt.hist(xt,bins=10,normed=True,alpha=0.1)

#muの推定 p(mu|x,sig2_r,mu_pr,sig2_pr)
mu_N,sig2_N = bayse(xt[0:1],sig2_r,mu_pr,sig2_pr)
plt.plot(x,gauss(x,mu_N,sig2_N))

mu_N,sig2_N = bayse(xt[0:2],sig2_r,mu_pr,sig2_pr)
plt.plot(x,gauss(x,mu_N,sig2_N))

mu_N,sig2_N = bayse(xt[0:TN],sig2_r,mu_pr,sig2_pr)
plt.plot(x,gauss(x,mu_N,sig2_N))

#期待値
plt.axvline(x=mu_r,c='gray')

plt.ylim([0,5])
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 22:34:46 2016

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt

def gauss_dist(x,mu,lam):
    ret= np.sqrt(lam/(2*np.pi)) * np.exp(-lam*(x-mu)**2/2)
    return ret

def mixture_gauss_dist(x,pi,sig1,mu1,sig2,mu2):
    ret=pi*gauss_dist(x,mu1,1/sig1**2)+(1-pi)*gauss_dist(x,mu2,1/sig2**2)
    return ret

def mixture_gauss(pi,sig1,mu1,sig2,mu2):
    a=np.random.mtrand.binomial(1,pi)
    if(a==1):
        xt=(np.random.randn(1)*sig1)+mu1
    else:
        xt=(np.random.randn(1)*sig2)+mu2
    return xt

def kern_hypercube(u):
    ret = 1
    if(np.fabs(u) > 0.5):
        ret = 0
    return ret

def density_estimation_hypercube(x,xt,h):
    sumk=0
    D=1
    N=xt.shape[0]
    for i in xt:
        sumk+=kern_hypercube((x-i)/h)
    ret=sumk/(N*h**D)
    return ret
    

def density_estimation_gauss(x,xt,h):
    sumk=0
    N=xt.shape[0]
    for i in xt:
        sumk+=gauss_dist(x,i,1/h**2)
    ret=sumk/N
    return ret

    
def density_estimation_knearest(x,xt,k):
    dist = 2*np.fabs(x-xt)
    N=xt.shape[0]
    v = np.sort(dist)[k-1]
    ret=k/(N*v)
    return ret

pi=0.3
sig1=0.1
mu1 = 0.3
sig2=0.1
mu2 = 0.7

#訓練データ生成
N=1000
xt=np.zeros(N)
for i in range(N):
    xt[i]=mixture_gauss(pi,sig1,mu1,sig2,mu2)

#プロット用データ
x=np.arange(0,1,0.01)
y=np.zeros(x.shape[0])
no=1
rows=2
cols=2

# ヒストグラム法
plt.subplot(rows,cols,no)
plt.plot(x,mixture_gauss_dist(x,pi,sig1,mu1,sig2,mu2))
plt.hist(xt, normed = True,bins=30)
no=no+1

# カーネル密度推定（超立方体カーネル）
plt.subplot(rows,cols,no)
h=0.1
plt.plot(x,mixture_gauss_dist(x,pi,sig1,mu1,sig2,mu2))
for i in range(x.shape[0]):
    y[i] = density_estimation_hypercube(x[i],xt,h)
plt.plot(x,y)
no=no+1

# カーネル密度推定（ガウスカーネル）
plt.subplot(rows,cols,no)
h=0.03
plt.plot(x,mixture_gauss_dist(x,pi,sig1,mu1,sig2,mu2))
for i in range(x.shape[0]):
    y[i] = density_estimation_gauss(x[i],xt,h)
plt.plot(x,y)
no=no+1

# K近傍法
plt.subplot(rows,cols,no)
k=100
plt.plot(x,mixture_gauss_dist(x,pi,sig1,mu1,sig2,mu2))
for i in range(x.shape[0]):
    y[i] = density_estimation_knearest(x[i],xt,k)
plt.plot(x,y)
no=no+1

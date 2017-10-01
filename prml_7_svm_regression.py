#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 09:40:04 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt

## カーネル関数
def kern(x1,x2):
    beta = 10
    ret = np.exp(-beta * (x1-x2)**2)
    return ret


# # グラム行列
def gramMat(x):
    size = x.shape[0]
    ret = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            ret[i,j] = kern(x[i],x[j])
    return ret


# # 元の関数
def realFunc(x):
    ret = np.sin(2*np.pi*x)
    return ret


# # コスト関数
def L(a,ahat,t,K,eps):
    size = a.shape[0]
    ret = 0
    for n in range(size):
        for m in range(size):
            ret += -0.5*(a[m] - ahat[m])*(a[n] - ahat[n]) * K[m,n]
        ret += - eps * (a[n] + ahat[n])
        ret += t[n] * (a[n] - ahat[n])
    return ret


# # 勾配
def dadL(a,ahat,t,K,eps):
    size = a.shape[0]
    ret_a = np.zeros(size)
    ret_ahat = np.zeros(size)
    for n in range(size):
        tmp = 0
        for m in range(size):
            tmp += (a[m] - ahat[m]) * K[n,m]
        ret_a[n] = - tmp + t[n] - eps
        ret_ahat[n] =  tmp - t[n] - eps
    return (ret_a,ret_ahat)
    
    


dataS = 10
xt = np.linspace(0, 1, dataS)
yt = realFunc(xt) + np.random.randn(dataS) * 0.5


# # aの最適化
eps = 0.5
a = np.zeros(dataS)
ahat = np.zeros(dataS)
K = gramMat(xt)
C = 10

learnMax = 10000
print("cost:",L(a,ahat,yt,K,eps))
for i in range(learnMax):
    eta = 0.05
    d_a,d_ahat = dadL(a,ahat,yt,K,eps)
    a += eta * d_a
    ahat += eta * d_ahat
    a = np.minimum(np.maximum(a, 0),C)
    ahat = np.minimum(np.maximum(ahat, 0),C)
print("cost:",L(a,ahat,yt,K,eps))
print("sum(a-a^):",(a-ahat).sum())

#
## # bの最適化(a+a^の合計が0であることを保証できないので、計算不可)
#ak = (a-ahat).dot(K)
#b = 0
#N = 0
#for n in range(dataS):
#    if(0 < a[n] and a[n] < C):
#        b += yt[n] - eps - ak[n]
#        N += 1
#    elif(0 < ahat[n] and ahat[n] < C):
#        b += yt[n] + eps - ak[n]
#        N += 1
#
#if(N != 0):
#    b /= N
#print("b:",b)



# # プロット
pltS = 100
x = np.linspace(0.0,1.0,pltS)
y = np.zeros(pltS)

for i in range(pltS):
    for n in range(dataS):
        y[i] += (a[n] - ahat[n]) * kern(xt[n],x[i])
plt.plot(x,y)
plt.plot(xt,yt,'o')
plt.plot(x,realFunc(x),'r')
plt.fill_between(x,y-eps,y+eps,alpha=0.5,facecolor='pink')


for n in range(dataS):
    if(a[n] != 0):
        plt.plot(xt[n],yt[n],'o',ms = 10,c='None',mec='magenta')
        
    if(ahat[n] != 0):
        plt.plot(xt[n],yt[n],'o',ms = 10,c='None',mec='blue')
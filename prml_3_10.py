#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 01:09:25 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#plt.clf()

def basisFunc(x,j,M):
    #ダミーの基底関数
    if(j==0):
        return np.ones(x.shape)
    mu=(j-1)*2/(M-1)-1
#    
#    sig2=1/M**2
#    ret=np.exp(-(x-mu)**2/(2*sig2))
    #多項式
#    ret = x**j
    #シグモイド
    a=(x-mu)/0.1
    ret = 1/(1+np.exp(-a))
    return ret
    
    
def PHI(x,M):
    N=x.shape[0]
    ret=np.zeros([N,M])
    for n in range(N):
        for m in range(M):
            ret[n,m]=basisFunc(x[n],m,M)
    return ret

def phi(x,M):
    ret=np.zeros(M)
    for m in range(M):
        ret[m]=basisFunc(x,m,M)
    return ret

def kern(x1,x2,M,beta):
    rows=x1.shape[0]
    cols=x2.shape[0]
    X1,X2 = np.meshgrid(x1, x2)
    Sn=np.linalg.inv(beta * PHI(x1,M).transpose().dot(PHI(x1,M)))
    ret = np.zeros([rows,cols])
    for i in range(rows):
        for j in range(cols):
            ret[i,j] = beta * phi(X1[i,j],M).transpose().dot(Sn).dot(phi(X2[i,j],M))
    return X1,X2,ret
    

beta= 0.01
M=11
x1=np.arange(-1, 1, 0.01)
x2=np.arange(-1, 1, 0.01)
X1,X2,K = kern(x1,x2,M,beta)
#plt.contourf(X1, X2, K,100)

ax = Axes3D(plt.figure())
ax.plot_surface(X1,X2,K, cmap=cm.jet, rstride=5, cstride=5)
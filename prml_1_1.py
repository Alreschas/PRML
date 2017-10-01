# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:30:44 2016

@author: Narifumi
"""
#多項式曲線フィッティング

import numpy as np
import matplotlib.pyplot as plt
plt.clf()

M=5

def y(x,w):
    ret=0
    for i in range(M+1):
        ret+=w[i]*(x**i)
    return ret

def calcWeight(t,x):
    tx=np.zeros([M+1,1])
    phi=np.zeros([M+1,M+1])
    for i in range(M+1):
        tx[i]=(t*(x**i)).sum()
    
    for i in range(M+1):
        for j in range(M+1):
            phi[i,j]=(x**(i+j)).sum()
    
    w=np.linalg.inv(phi).dot(tx)
    return w        


data = np.loadtxt('data/curvefitting.txt', comments='#' ,delimiter=' ')
x=data[:,0]
t=data[:,1]

N=100
sig2=0.1
L=1000

w_ml=np.zeros([M+1,1])
sig2_ml=0

for l in range(L):
    
    x=np.random.rand(N)
    t=np.sin(2*np.pi*x)+np.random.randn(N)*np.sqrt(sig2)

    #教師データの描画
#    plt.plot(x,t,'o',ms=10,alpha=0.2,c='none',markeredgecolor='blue',markeredgewidth=1)

    #多項式の係数を求める
    w=calcWeight(t,x)

    #分散を求める
    w_ml+=w/L
    sig2_ml+=(((t-y(x,w))**2).sum(0)/(N*L))


#近似曲線の描画
x=np.arange(0,1.01,0.01)
plt.plot(x,y(x,w_ml),'r', ms=15, lw=1, alpha=1.0)
xs = np.concatenate( (x,x[::-1]) )
ys = np.concatenate( (y(x,w_ml)+np.sqrt(sig2_ml),(y(x,w_ml)-np.sqrt(sig2_ml))[::-1]))
p = plt.fill(xs, ys, facecolor='pink', alpha=0.5,lw=0)



#元の関数の描画
y=np.sin(2*np.pi*x)
plt.plot(x,y,'g', ms=15, lw=1, alpha=1.0)
plt.plot(x,y+np.sqrt(sig2),'g--', ms=15, lw=1, alpha=0.5)
plt.plot(x,y-np.sqrt(sig2),'g--', ms=15, lw=1, alpha=0.5)

plt.xlim(-0.1, 1.1)
plt.ylim(-1.5, 1.5)
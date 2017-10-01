#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:51:42 2016

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt

plt.clf()

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,mu,sig2):
    p = (1/np.sqrt(2*np.pi*sig2))*np.exp(-(x-mu)**2/(2*sig2))
    return p
    
beta = 11.1
alpha= 5.0e-3

sig2=0.1
data = np.loadtxt('data/curvefitting.txt', comments='#' ,delimiter=' ')
x_t=data[:,0]
y_t=data[:,1]

#訓練データ
#N=10
#x_t = np.random.rand(N)
#y_t=np.sin(x_t*2*np.pi)+np.sqrt(sig2) * np.random.randn(N)


N=x_t.shape[0]


def basisFunc(x,j,M):
    #ダミーの基底関数
    if(j==0):
        return np.ones(x.shape)

    if(M == 2):
        mu = 0.5
    else:
        mu=(j-1)/(M-2)
    
    sig2=1/M**2
    ret=np.exp(-(x-mu)**2/(2*sig2))
    return ret

def phi(x,M):
    ret=np.zeros([M,x.shape[0]])
    for i in range(M):
        ret[i,:]=basisFunc(x,i,M)
    return ret


def calcEvidence(x_t,y_t,M):   
    N = x_t.shape[0]
    y_t = np.matrix(y_t).transpose()
    phix=phi(x_t,M).transpose()
    phixt2=np.zeros([M,M])
    phixtxtn=np.zeros([M,1])
    for i in range(N):
        phixtxtn+=phi(x_t,M)[:,i:i+1]*y_t[i]#ΦTt
        phixt2+=phi(x_t,M)[:,i:i+1].dot(phi(x_t,M)[:,i:i+1].T) #ΦTΦ
    SNinv=alpha*np.identity(M)+beta*phixt2
    SN=np.linalg.inv(SNinv)#SN
    mN=beta*SN.dot(phixtxtn)
    EmN = 0.5 * alpha * mN.transpose().dot(mN)
    EmN += 0.5 * beta * (y_t - (phix.dot(mN))).transpose().dot(y_t - (phix.dot(mN)))
    evidence = 0.5*M*np.log(alpha)
    evidence += 0.5*N*np.log(beta)
    evidence -= EmN
    evidence -= 0.5 * np.log(np.linalg.det(SNinv))
    evidence -= 0.5 * N * np.log(2*np.pi)
    return evidence

#基底関数の数
plt.subplot(2,1,1)
M = np.arange(10)
evidence=np.zeros(M.shape[0])
for m in M:
    evidence[m] = calcEvidence(x_t,y_t,m+1)
plt.plot(M,evidence,'-')

plt.subplot(2,1,2)
x=np.arange(0,1.0,0.01)
#xs = np.concatenate( (x,x[::-1]) )
#ys = np.concatenate( (m+np.sqrt(s2),(m-np.sqrt(s2))[::-1]) )
#p = plt.fill(xs, ys, facecolor='pink', alpha=0.5,lw=0)

#基底関数
for i in range(5):
    plt.plot(x,basisFunc(x,i,5))

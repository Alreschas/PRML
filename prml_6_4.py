#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:00:25 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussKern(x1,x2):
    beta = 10
    ret = np.exp(-beta*(x1-x2)**2)
    return ret

def expKern(x1,x2):
    theta = 10
    ret = np.exp(-theta*np.abs(x1-x2))
    return ret

def gramMat(x,kern):
    size = x.shape[0]
    ret = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            ret[i,j] = kern(x[i],x[j])
            
    return ret

fig = plt.figure()

dataS = 200
x = np.linspace(-1,1,dataS)
mu = np.zeros(dataS)

#ガウスカーネル
np.random.seed(0)
fig.add_subplot(211)
K = gramMat(x,gaussKern)
for i in range(2):
    y=np.random.multivariate_normal(mu,K)
    plt.plot(x,y)
    
#tの事前分布
beta = 10
np.random.seed(0)
fig.add_subplot(211)
K = gramMat(x,gaussKern)
for i in range(2):
    y=np.random.multivariate_normal(mu,K+np.eye(dataS)/beta)
    plt.plot(x,y,'.',alpha = 0.5)

#指数カーネル   
np.random.seed(0)
fig.add_subplot(212)
K = gramMat(x,expKern)
for i in range(2):
    y=np.random.multivariate_normal(mu,K)
    plt.plot(x,y)

beta = 100
np.random.seed(0)
fig.add_subplot(212)
K = gramMat(x,expKern)
for i in range(2):
    y=np.random.multivariate_normal(mu,K+np.eye(dataS)/beta)
    plt.plot(x,y,'.',alpha=0.5)   
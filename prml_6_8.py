#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:05:51 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def realFunc(x):
    ret = np.sin(x*2*np.pi)
    return ret

def kern(x1,x2,th1):
    ret = np.exp(-0.5*th1 * (x1-x2)**2)
#    theta0 = 1
#    theta1 = 10
#    theta2 = 10
#    theta3 = 0
#    ret = theta0 *np.exp(-0.5*theta1*(x1-x2)**2) + theta2 + theta3*x1*x2
    return ret

def dkern_th1(x1,x2,th1):
    ret = (-0.5 * (x1-x2)**2) * np.exp(-0.5*th1 * (x1-x2)**2)
    return ret

def gramMatBB(x,beta,th1):
    dataS = x.shape[0]
    ret = np.zeros([dataS,dataS])
    for i in range(dataS):
        for j in range(dataS):
            ret[i,j] = kern(x[i],x[j],th1)
    ret += np.eye(dataS)/beta
    return ret

def dgramMatBB_th1(x,th1):
    dataS = x.shape[0]
    ret = np.zeros([dataS,dataS])
    for i in range(dataS):
        for j in range(dataS):
            ret[i,j] = dkern_th1(x[i],x[j],th1)
    return ret


def J(theta, *args):
    xt,yt = args
    beta,th1 = theta
    dataS = xt.shape[0]
    CN = gramMatBB(xt,beta,th1)
    invCN = np.linalg.inv(CN)

    ret = -0.5*np.log(np.linalg.det(CN)) - 0.5*yt.T.dot(invCN).dot(yt) - 0.5 * dataS * np.log(2*np.pi)
    return -ret

    
def dlnp(theta, *args):
    xt,yt = args
    beta,th1 = theta
    CN = gramMatBB(xt,beta,th1)
    invCN = np.linalg.inv(CN)
    dCN_beta = dgramMatBB_beta(xt,beta)
    dCN_th1 = dgramMatBB_th1(xt,beta)
    ret = np.zeros(2)
  
    ret[0] = -0.5 * np.trace(invCN.dot(dCN_beta)) + 0.5 * yt.T.dot(invCN).dot(dCN_beta).dot(invCN).dot(yt)
    ret[1] = -0.5 * np.trace(invCN.dot(dCN_th1)) + 0.5 * yt.T.dot(invCN).dot(dCN_th1).dot(invCN).dot(yt)
    
    return -ret

def dgramMatBB_beta(x,beta):
    dataS = x.shape[0]
    ret = -np.eye(dataS)/(beta**2)
    return ret


def gramMatAB(x,xt,th1):
    dataS = xt.shape[0]
    ret = np.zeros(dataS)
    for i in range(dataS):
        ret[i] = kern(x,xt[i],th1)
    return ret
            
plotS = 100
sigt = 0.3
#xt = np.random.rand(plotS)
xt = np.linspace(0,1,plotS)
yt = realFunc(xt) +np.random.randn(plotS) * sigt
plt.plot(xt,yt,'o')

tgtS = 70
xt = xt[0:tgtS]
yt = yt[0:tgtS]
plt.plot(xt,yt,'o')

dataS = 100
x = np.linspace(0.0,1.,dataS)
y = realFunc(x)
plt.plot(x,y,'-.')

print(1/(sigt**2))
beta = 25
th1 = 10

args = (xt,yt)
init_param = np.array([beta, th1])
param = optimize.fmin_cg(J, init_param,fprime=dlnp, args=args)
beta = param[0]
th1 = param[1]

CN = gramMatBB(xt,beta,th1)
invCN = np.linalg.inv(CN)

mu = np.zeros(dataS)
var = np.zeros(dataS)
print(beta,th1)
for i in range(dataS):    
    k = gramMatAB(x[i],xt,th1)
    c = kern(x[i],x[i],th1) + 1/beta
    mu[i] = k.T.dot(invCN).dot(yt)
    var[i] = c - k.T.dot(invCN).dot(k)
plt.plot(x,mu)
plt.ylim([-1.5,1.5])
plt.fill_between(x,mu-2*np.sqrt(var),mu+2*np.sqrt(var),facecolor='pink',alpha=0.5,lw=0)

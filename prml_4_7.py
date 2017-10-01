#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 21:23:21 2017

@author: Narifumi
"""
#パーセプトロンアルゴリズム
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random
clf()

c_red = 1
c_blue = -1

def makeData(cls):
    x = -0.5 + np.random.random([2,1])
    if(cls == c_red):
        while(-1*x[0] > x[1]-0.1):
            x = -0.5+np.random.random([2,1])
    else:
        while(-1*x[0] < x[1]+0.1):
            x = -0.5+np.random.random([2,1])  
    return x

def y(x,w):
    a = w.T.dot(x)
    ret = 1
    if(a<0):
        ret = -1
    return ret

def learn(x,w,eta,cls):
    x2 = np.zeros([3,1])
    x2[0] = 1
    x2[1:3,:] = x
    if(y(x2,w) != cls):
        w = w + eta * x2 * cls
    return w

def plot(w,c,a):
    x = np.linspace(-0.5, 0.5, 100)
    y = - w[1,0]/w[2,0] * x - w[0,0]/w[2,0]
    plt.plot(x,y,'-',color=c,alpha=a)

w = np.matrix([[2],[-5],[-3]])
plot(w,'r',0.5)

#学習
eta = 1
for i in range(100):
    x = makeData(c_red)
    plt.scatter(x[0],x[1],color='r',alpha=0.5,s=5.0)
    w = learn(x,w,eta,c_red)
        
    x = makeData(c_blue)
    plt.scatter(x[0],x[1],color='b',alpha=0.5,s=5.0)
    w = learn(x,w,eta,c_blue)

    if(i % 5 == 0):
        plot(w,'b',0.2)


plot(w,'g',1)
#plt.xlim(-0.5,0.5)
plt.axis('equal')
plt.ylim(-0.7,0.7)


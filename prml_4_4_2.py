#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:41:34 2017

@author: Narifumi
"""

#確率的勾配法によるロジスティック回帰

import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random

plt.clf()


def gaussian(x,mu,sig):
    D=2
    ret = np.exp(-0.5*(x-mu).T.dot(np.linalg.inv(sig)).dot(x-mu))
    ret = ret/((2*np.pi)**(D/2) * np.sqrt(np.linalg.det(sig)))
    return ret

def sigmoid(a):
    ret= 1/(1+exp(-a))
    return ret

def softmax(a,b,c):
    ret= np.exp(a)/(np.exp(a)+np.exp(b)+np.exp(c))
    return ret

def learn_ml(x,t):
    w=np.matrix(np.random.rand(3)).T
    eta=0.1
    for n in range(N):
        xn = np.matrix(np.append([1],x[n])).T
        tn = t[n,0]
        yn=sigmoid(w.T.dot(xn)[0,0])
        divE = (yn-tn)*xn
        w=w-eta*divE
    return w

xmax=10
xmin=-10
ymax=10
ymin=-10

#訓練データ
N_red = 100
mu_red = [-3,3]
cov_red =   [[0.6,0], [0.0,0.4]]

N_blue = 100
mu_blue = [-1,1]
cov_blue =  [[0.6,0], [0.0,0.4]]

#外れ値
N_b_out = 0
mu_b_out= [7,-7]
cov_out = [[1,0], [0,1]]


#ガウス分布
x_red,y_red = np.random.multivariate_normal(mu_red,cov_red,N_red).T
x_blue,y_blue = np.random.multivariate_normal(mu_blue,cov_blue,N_blue).T

#外れ値の追加
x_b_out,y_b_out = np.random.multivariate_normal(mu_b_out,cov_out,N_b_out).T
x_blue, y_blue = np.r_[x_blue,x_b_out], np.r_[y_blue,y_b_out]
N_blue+=N_b_out

#データをまとめる
x = vstack((hstack((x_red, x_blue)).T,hstack((y_red,y_blue)).T)).T
t = np.matrix([[1,0]]*N_red+[[0,1]]*N_blue)
N=N_red+N_blue

#プロット用データ
x_line = np.linspace(xmin, xmax, 100)
y_line = np.linspace(ymin, ymax, 100)
X, Y = meshgrid(x_line, y_line)



def f(x,w):
    ret = -(w[0,0]/w[2,0]+w[1,0]/w[2,0]*x)
    return ret

for time in range(10):

    w=learn_ml(x,t)
    plt.plot(x_line,f(x_line,w),'k')
    #結果のプロット

#Z=np.zeros(X.shape)
#for i in range(X.shape[0]):
#    for j in range(X.shape[1]):
#        x=np.matrix([X[i,j],Y[i,j]]).T
#        Z[i,j] = sigmoid(w.T.dot(x))
##plt.contour(X,Y,Z)


plt.scatter(x_red,y_red,color='r',marker='x')
plt.scatter(x_blue,y_blue,color='none',marker='o',edgecolors='b')


xlim(xmin, xmax)
ylim(ymin, ymax)

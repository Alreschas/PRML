#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 21:30:08 2017

@author: Narifumi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 21:44:27 2017

@author: Narifumi
"""


import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random

plt.clf()

xmin=-2
xmax=2
ymin=-2
ymax=2

#訓練データ
N_red = 100
N_green = 100
N_blue = 100
mu_red = [-1,1]
mu_green = [1,0]
mu_blue = [-1,-1]

def makeData(cls,num):
    retx=np.zeros(num)
    rety=np.zeros(num)
    for i in range(num):
        x = -0.5 + np.random.random([2,1])
        if(cls == 0):
            while(-1*x[0] > x[1]-0.1):
                x = -0.5+np.random.random([2,1])
        else:
            while(-1*x[0] < x[1]+0.1):
                x = -0.5+np.random.random([2,1])
        retx[i] = x[0]
        rety[i] = x[1]
    return retx,rety

cov_red =   [[0.15,0], [0.0,0.1]]
cov_blue =  [[0.15,0], [0.0,0.1]]
cov_green = [[0.05,0], [0.0,0.5]]
#ガウス分布の場合
x_red,y_red = np.random.multivariate_normal(mu_red,cov_red,N_red).T
x_blue,y_blue = np.random.multivariate_normal(mu_blue,cov_blue,N_blue).T
x_green,y_green = np.random.multivariate_normal(mu_green,cov_green,N_green).T

#ガウス分布でない場合
#x_red,y_red = makeData(0,N_red)
#x_blue,y_blue = makeData(1,N_blue)
#x_green,y_green = np.random.multivariate_normal(mu_green,cov_green,N_green).T

x = vstack((hstack((x_red, x_blue,x_green)).T,hstack((y_red,y_blue,y_green)).T)).T
t = np.matrix([[1,0,0]]*N_red+[[0,1,0]]*N_blue+[[0,0,1]]*N_green)
N=N_red+N_blue+N_green

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


pi1=N_red/(N)
pi2=N_blue/(N)
pi3=N_green/(N)
mu1=(t[:,0].T*x/N_red).T
mu2=(t[:,1].T*x/N_blue).T
mu3=(t[:,2].T*x/N_green).T

s1=0
s2=0
s3=0
for i in range(N):
    xtmp=np.matrix(x[i]).T
    s1+=(xtmp-mu1).dot((xtmp-mu1).T)*t[i,0]/N_red
    s2+=(xtmp-mu2).dot((xtmp-mu2).T)*t[i,1]/N_blue
    s3+=(xtmp-mu3).dot((xtmp-mu3).T)*t[i,2]/N_green
s=N_red/N*s1 + N_blue/N*s2 + N_green/N*s3

x_line = np.linspace(xmin, xmax, 100)
y_line = np.linspace(ymin, ymax, 100)
X, Y = meshgrid(x_line, y_line)
plt.scatter(x_red,y_red,color='r',marker='x')
plt.scatter(x_green,y_green,color='g',marker='+')
plt.scatter(x_blue,y_blue,color='none',marker='o',edgecolors='b')

Z1=np.zeros(X.shape)
Z2=np.zeros(X.shape)
Z3=np.zeros(X.shape)
Z=np.zeros(X.shape)
pc1=np.zeros(X.shape)
pc2=np.zeros(X.shape)
pc3=np.zeros(X.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xtmp=np.matrix([[X[i,j]],[Y[i,j]]])
        Z1[i,j] = gaussian(xtmp,mu1,s1)
        Z2[i,j] = gaussian(xtmp,mu2,s2)
        Z3[i,j] = gaussian(xtmp,mu3,s3)
        #sが共通の場合は、線形分離可能
#        Z1[i,j] = gaussian(xtmp,mu1,s)
#        Z2[i,j] = gaussian(xtmp,mu2,s)
#        Z3[i,j] = gaussian(xtmp,mu3,s)
        a1=np.log(pi1*Z1[i,j])
        a2=np.log(pi2*Z2[i,j])
        a3=np.log(pi3*Z3[i,j])
        pc1[i,j] = softmax(a1,a2,a3)
        pc2[i,j] = softmax(a2,a3,a1)
        pc3[i,j] = softmax(a3,a1,a2)
        
plt.contour(X,Y,Z1,alpha=0.3)
plt.contour(X,Y,Z2,alpha=0.3)
plt.contour(X,Y,Z3,alpha=0.3)
plt.contour(X,Y,pc1,alpha=0.3)
plt.contour(X,Y,pc2,alpha=0.3)
plt.contour(X,Y,pc3,alpha=0.3)


xlim(xmin, xmax)
ylim(ymin, ymax)

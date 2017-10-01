#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#ガウス分布の密度関数
def gaussianDist(sig,mu,x):
    y=np.exp(-(x-mu)**2/(2*sig**2))/(np.sqrt(2*np.pi)*sig)
    return y

#確率変数の変換関数
def g(y):
    x=np.log(y)-np.log(1-y)+5
    return x
    
#確率変数の変換関数の逆関数
def invg(x):
    y=1/(1+np.exp(-x+5))
    return y

#ガウス分布px(x)の平均、分散
sig=1.0
mu=6

#ヒストグラムのサンプル数
N = 50000 

plt.xlim([0,10])
plt.ylim([0,1])

####
x = np.linspace(0,10,100)

#確率変数の変換関数をプロット
y=invg(x)
plt.plot(x,y,'b')

#px(x)のプロット
y = gaussianDist(sig,mu,x)
plt.plot(x,y,'r')

#px(x)からのサンプルを元にヒストグラムをプロット
x_sample = mu + sig * np.random.randn(N)
plt.hist(x_sample,bins=20,normed=True,color='lavender')


####
y=np.linspace(0.01,0.99,100)

##py(y)のプロット
x=gaussianDist(sig,mu,g(y))/(y*(1-y))
plt.plot(x,y,'m')

#px(x)からのサンプルをg^-1(x)で変換したデータのヒストグラムをプロット
y_sample = invg(mu + sig * np.random.randn(N))
plt.hist(y_sample,bins=20,normed=True,orientation="horizontal",color='lavender')

#px(g(y))のように単純に変換した関数をプロット
x = gaussianDist(sig,mu,g(y))
plt.plot(x/(x.sum()*0.01) ,y,'lime')

####
#平均muとg^-1(mu)との関係をプロット
plt.plot([mu, mu], [0, invg(mu)], 'k--')
plt.plot([0, mu], [invg(mu), invg(mu)], 'k--')

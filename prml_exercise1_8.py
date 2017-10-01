#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#プロット用データ
X=np.arange(1,5,0.001)

def gaussian(x,mu,sig2):
    y=np.exp(((x-mu)**2)/(-2*sig2))/np.sqrt(2*np.pi*sig2)
    return y

#真の平均・分散
mu=3
sig2=1.0

#一回の最尤推定に利用するデータ数
N=2

#最尤推定を行う回数
M=10000

#N個を一セットとして、Mセットのデータを真の分布から生成
x=np.sqrt(sig2)*np.random.randn(N,M)+mu

#N個のデータを利用した最尤推定を、Mセット分実施
mu_ml=(1/N)*x.sum(0)
sig2_ml=(1/N)*((x-mu_ml)**2).sum(0)

#真の分布
plt.plot(X,gaussian(X,mu,sig2),'r',lw=15,alpha=0.5)

#M回の最尤推定の結果を平均した分布
plt.plot(X,gaussian(X,mu_ml.sum(0)/M,sig2_ml.sum(0)/M),'g')

#最尤推定値の分散をN/N-1倍して、バイアスを取り除いたもの
plt.plot(X,gaussian(X,mu_ml.sum(0)/M,sig2_ml.sum(0)/M*N/(N-1)),'b')

#plt.savefig("Users/Narifumi/Desktop/test.png")
plt.savefig("/Users/Narifumi/Desktop/test.png")

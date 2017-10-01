#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:52:30 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

plt.clf()

def aFunc_l1(x):
    ret = np.tanh(x)
    return ret
def aFunc_l2(x):
    ret = x
    return ret
def out_l1(x,w_l1):
    ret=np.matrix(np.append(1,aFunc_l1(w_l1.T.dot(x)))).T
    return ret
def out_l2(x,w_l2):
    ret=np.matrix(aFunc_l2(w_l2.T.dot(x))).T
    return ret

#正則化コスト関数
def J2(param, *args):
    """最小化を目指すコスト関数を返す"""
    w = param
    wl1=w[:(N_in+1)*N_hdn].reshape(N_in+1,N_hdn)
    wl2=w[(N_in+1)*N_hdn:].reshape(N_hdn+1,N_out)
    # パラメータ以外のデータはargsを通して渡す
    xt,yt,a,c = args
    N=xt.shape[0]
    E=0
    lam1 = a**2*1
    lam2 = 1/(c**2)
    for n in range(N):
        z_l0 = np.matrix(np.append(1,xt[n])).T
        z_l1 = out_l1(z_l0,wl1)
        z_l2=out_l2(z_l1,wl2)
        y=z_l2[0,0]
        E+=(y-yt[n])**2
    E = E/2 / (c**2) + lam1*(w[N_hdn:(N_in+1)*N_hdn]**2).sum(0)/2 + lam2*(w[(N_in+1)*N_hdn+N_out:]**2).sum(0)/2
    return E

def plot(w,xd):
    #プロット
    yd = np.zeros(xd.shape)
    w_l1=w[:(N_in+1)*N_hdn].reshape(N_in+1,N_hdn)
    w_l2=w[(N_in+1)*N_hdn:].reshape(N_hdn+1,N_out)

    for i in range(xd.shape[0]):
        z_l0 = np.matrix(np.append(1,xd[i])).T
        z_l1 = out_l1(z_l0,w_l1)
        z_l2=out_l2(z_l1,w_l2)
        yd[i]= z_l2[0,0]
    plt.plot(xd,yd)
    return np.max(yd),np.min(yd)

def addReg(w,alphaW1,alphaB1,alphaW2,alphaB2):
    w2=copy.deepcopy(w)
    w2[:N_hdn]/=np.sqrt(alphaB1)
    w2[N_hdn:(N_in+1)*N_hdn]/=np.sqrt(alphaW1)

    w2[(N_in+1)*N_hdn:(N_in+1)*N_hdn+N_in]/=np.sqrt(alphaB2)
    w2[(N_in+1)*N_hdn+N_in:]/=np.sqrt(alphaW2)
    return w2
    

xd = np.arange(-1.,1.,0.001)
x_t=np.zeros(10)
y_t=np.zeros(10)
#学習パラメータ
N_out = 1
N_in = 1
N_hdn = 12

#無矛盾の検証
w = np.random.randn((N_in+1)*N_hdn+(N_hdn+1)*N_out)*np.sqrt(100)
w1=copy.deepcopy(w)
w2=copy.deepcopy(w)
w3=copy.deepcopy(w)
a=1/np.sqrt(5)
b=0.1
c=10
d=100

w2[:N_hdn]-=(b/a)*w3[N_hdn:(N_in+1)*N_hdn]
w2[N_hdn:(N_in+1)*N_hdn]/=a

w3[(N_in+1)*N_hdn:]*=c
w3[(N_in+1)*N_hdn:(N_in+1)*N_hdn+N_out]+=d


plt.subplot(4,2,1)
plt.title("orginal")
k,j=plot(w1,xd)
plt.text(-0.9, k-(k-j)/10, "Err:%.2f" % J2(w1,*(x_t,y_t,1,1)))

plt.subplot(4,2,3)
plt.title("scale input")
k,j=plot(w2,xd)
plt.text(-0.9, k-(k-j)/10, "Err:%.2f" % J2(w2,*(x_t*a+b,y_t,a,1)))

plt.subplot(4,2,5)
plt.title("scale output")
k,j=plot(w3,xd)
plt.text(-0.9, k-(k-j)/10, "Err:%.2f" % J2(w3,*(x_t,y_t*c+d,1,c)))


#print(J2(w1,*(x_t,y_t,1,1)))
#print(J2(w2,*(x_t*a+b,y_t,a,1)))
#print(J2(w3,*(x_t,y_t*c+d,1,c)))

#図5.11
param=[ (1    ,1    ,1    ,1),
        (1    ,1    ,1e-2 ,1),#W2は縦のスケール
        (1e-6 ,1e-4 ,1    ,1),#W1は横のスケール
        (1e-6 ,1e-6 ,1    ,1),#B1は変化の範囲
        (1    ,1    ,1    ,1e-2)]#B2は縦の位置
for j in range(5):
    w = np.random.randn((N_in+1)*N_hdn+(N_hdn+1)*N_out)*np.sqrt(1)
    for i in range(len(param)-1):
        plt.subplot(4,2,(i+1)*2)
        w2=addReg(w,param[i][0],param[i][1],param[i][2],param[i][3])
        plot(w2,xd)

plt.show()
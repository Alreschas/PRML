#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:07:15 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

plt.clf()

ims = []

def gaussian(x,mu,sig2):
    p = (1/np.sqrt(2*np.pi*sig2))*np.exp(-(x-mu)**2/(2*sig2))
    return p

data = np.loadtxt('data/curvefitting.txt', comments='#' ,delimiter=' ')
x_t=data[:,0]
y_t=data[:,1]
N=10

#N=50
#x_t=np.linspace(0, 1, N).reshape(N, 1)
#y_t=np.sin(x_t*4*np.pi)
#np.random.shuffle(x_t)
#y_t=x_t * x_t
#y_t=np.sign(x_t)
#y_t = np.abs(x_t)


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

N_out = 1
N_in = 1
N_hdn = 1

#プロット用データ
xd = np.arange(-0.5,1.5,0.01)
yd = np.zeros(xd.shape)
zd = np.zeros([xd.shape[0],N_hdn+1])

#w_l1=(np.random.rand(N_in+1,N_hdn)-0.5)*0.01
#w_l2=(np.random.rand(N_hdn+1,N_out)-0.5)*0.01
w_l1=(np.random.randn(N_in+1,N_hdn))*np.sqrt(10)
w_l2=(np.random.randn(N_hdn+1,N_out))*np.sqrt(10)
#w_l1=np.zeros([N_in+1,N_hdn])
#w_l2=np.zeros([N_hdn+1,N_out])
y_all=[]
z_all=[]

E=15
eta=0.1
p=0
while(E>0.01 and p<5000):
    p+=1
    E=0
    dE1=np.zeros(w_l1.shape)
    dE2=np.zeros(w_l2.shape)
    for n in range(N):
        #forward prop
        z_l0=np.matrix(np.append(1,x_t[n])).T
        z_l1=out_l1(z_l0,w_l1)
        z_l2=out_l2(z_l1,w_l2)        
        y=z_l2
            
#        dE_l2_2 = np.zeros(w_l2.shape)
#        for i in range(dE_l2_2.shape[0]):
#            for j in range(dE_l2_2.shape[1]):            
#                w_l2_1=np.copy(w_l2)
#                w_l2_2=np.copy(w_l2)
#                eps=0.01
#                w_l2_1[i,j] = w_l2[i,j] + eps
#                w_l2_2[i,j] = w_l2[i,j] - eps
#                y2_1=np.matrix(out_l2(z_l1,w_l2_1)).T
#                y2_2=np.matrix(out_l2(z_l1,w_l2_2)).T
#                dE_l2_2[i,j] = ((y2_1-y_t[n])**2-(y2_2-y_t[n])**2)/2/(2*eps)
#    
#                
#        dE_l1_2 = np.zeros(w_l1.shape)
#        for i in range(dE_l1_2.shape[0]):
#            for j in range(dE_l1_2.shape[1]):
#                w_l1_1=np.copy(w_l1)
#                w_l1_2=np.copy(w_l1)
#                eps=0.01
#                w_l1_1[i,j] = w_l1[i,j] + eps
#                w_l1_2[i,j] = w_l1[i,j] - eps
#                z_l1_1=np.matrix(out_l1(z_l0,w_l1_1)).T
#                z_l1_2=np.matrix(out_l1(z_l0,w_l1_2)).T
#                y2_1=np.matrix(out_l2(z_l1_1,w_l2)).T
#                y2_2=np.matrix(out_l2(z_l1_2,w_l2)).T
#                dE_l1_2[i,j] = ((y2_1-y_t[n])**2-(y2_2-y_t[n])**2)/2/(2*eps)

        
        delta_l2 = np.matrix(y-y_t[n]).T    
        dE_l2 = z_l1.dot(delta_l2.T)
#        w_l2 = w_l2 - eta * dE_l2        

        delta_l1 = np.multiply(1-np.power(z_l1,2),w_l2.dot(delta_l2))
        dE_l1 = z_l0.dot(delta_l1[1:,0].T)
        #w_l1 = w_l1 - eta * dE_l1

        dE1+=dE_l1
        dE2+=dE_l2
#   バッチ学習
    w_l1 = w_l1 - eta * dE1/100
    w_l2 = w_l2 - eta * dE2/100
    
    #２乗和誤差の算出
    for n in range(N):
        z_l0 = np.matrix(np.append(1,x_t[n])).T
        z_l1 = out_l1(z_l0,w_l1)
        z_l2=out_l2(z_l1,w_l2)
        y=z_l2[0,0]
        E+=(y-y_t[n])**2/2


#        print("\n")
#        print("diff Layer1:\n",dE_l1 - dE_l1_2)
#        print("diff Layer2:\n",dE_l2 - dE_l2_2)
        
    if(p%100 == 0):
#        eta = np.sqrt(E[0])/40
        print(p,E)
        for i in range(xd.shape[0]):
            z_l0 = np.matrix(np.append(1,xd[i])).T
            z_l1 = out_l1(z_l0,w_l1)
            z_l2=out_l2(z_l1,w_l2)
            yd[i]= z_l2[0,0]
            zd[i]= z_l1.T
        y_all.append(copy.deepcopy(yd))
        z_all.append(copy.deepcopy(zd))

def plot(data):
    plt.cla()                      # 現在描写されているグラフを消去
    time=data%len(y_all)
    plt.title('time=' + str(time))

    #トレーニングデータのプロット
    plt.plot(x_t,y_t,'o')

    plt.plot(xd,y_all[time],'r')

    #隠れユニットの出力
    for i in range(N_hdn+1):
        plt.plot(xd,z_all[time][:,i],'--')
    plt.ylim([-2,2])

fig = plt.figure()
ani = animation.FuncAnimation(fig, plot, interval=10)
plt.show()
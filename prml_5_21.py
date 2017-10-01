#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:44:48 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation


def realFunc(x):
    ret = x + 0.3 * np.sin(2*np.pi * x)
    return ret

class neuralNetwork_mg:
    #K:混合数
    def __init__(self,Nin,Nhdn,K):
        self.Nin=Nin
        self.Nhdn=Nhdn
        self.K = K
        self.Npi = K
        self.Nsig = K
        self.Nmu = K*Nin
        self.Nout = self.Npi + self.Nsig + self.Nmu
        
        self.w1 = np.random.randn(Nin+1,Nhdn) * 0.1
        self.w2 = np.random.randn(Nhdn+1,self.Nout) * 0.1
        
        self.gamma = np.zeros(self.K).reshape(self.K,1)
        
        self.momentum = 0.9
        
        self.dEdw1 = np.zeros(self.w1.shape)
        self.dEdw2 = np.zeros(self.w2.shape)
        self.dEdw1_acc = np.zeros(self.w1.shape)
        self.dEdw2_acc = np.zeros(self.w2.shape)
        self.cost = np.zeros(1)
        self.ite=0
        
    def calcCost(self):
        self.cost[self.ite] += -np.log(self.mixG)
    
    def gauss(self,t,mu,sig):
        sig2 = np.power(sig,2)
        ret = np.exp(-((t-mu)**2)/(2*sig2)) / np.sqrt(2*np.pi*sig2)
        return ret
        
    def mixtureGauss(self,t,mu,sig,pi):
        ret = 0
        normal = self.gauss(t,mu,sig)
        gam =  pi * normal
        self.mixG = gam.sum()
        self.gamma = gam/self.mixG
        ret = self.mixG
        return ret

    def forward_test2(self,x):
        self.z1 = np.append(1,x).reshape(self.Nin+1,1)
        
        a2 = np.dot(self.w1.T,self.z1)
        self.z2 = np.append(1,np.tanh(a2)).reshape(self.Nhdn+1,1)
        
        a3 = np.dot(self.w2.T,self.z2)
        
        api,amu,asig = np.split(a3, [self.Npi, self.Npi+self.Nmu], axis=0)
        
        self.zpi = np.exp(api)/(np.exp(api).sum())
        self.zmu = amu
        self.zsig = np.exp(asig)
        return self.zpi,self.zmu

    def forward_test(self,x,y):
        self.z1 = np.append(1,x).reshape(self.Nin+1,1)
        
        a2 = np.dot(self.w1.T,self.z1)
        self.z2 = np.append(1,np.tanh(a2)).reshape(self.Nhdn+1,1)
        
        a3 = np.dot(self.w2.T,self.z2)
        
        api,amu,asig = np.split(a3, [self.Npi, self.Npi+self.Nmu], axis=0)
        
        self.zpi = np.exp(api)/(np.exp(api).sum())
        self.zmu = amu
        self.zsig = np.exp(asig)
        
        ret = self.mixtureGauss(y,self.zmu,self.zsig,self.zpi)                
        return ret

    def forward(self,x,y):
        self.z1 = np.append(1,x).reshape(self.Nin+1,1)
        
        a2 = np.dot(self.w1.T,self.z1)
        self.z2 = np.append(1,np.tanh(a2)).reshape(self.Nhdn+1,1)
        
        a3 = np.dot(self.w2.T,self.z2)
        
        api,amu,asig = np.split(a3, [self.Npi, self.Npi+self.Nmu], axis=0)
        
        self.zpi = np.exp(api)/(np.exp(api).sum())
        self.zmu = amu
        self.zsig = np.exp(asig)
        
        ret = self.mixtureGauss(y,self.zmu,self.zsig,self.zpi)
        self.calcCost()
                
        return ret
        
    def backward(self,yt):
        
        deltapi = self.zpi - self.gamma
        deltamu = self.gamma * ((self.zmu - yt)/(self.zsig**2))
        deltasig = self.gamma * (self.Nin - ((yt - self.zmu)/self.zsig)**2)
        
        delta3 = np.vstack([deltapi,deltamu,deltasig])

        self.dEdw2 += np.dot(self.z2,delta3.T)
        
        delta2 = (1-self.z2**2) * np.dot(self.w2,delta3)
        self.dEdw1 += np.dot(self.z1,delta2[1:,:].T)
    
    def updateWeight(self,eta):
        self.dEdw1_acc = eta * self.dEdw1 + self.momentum * self.dEdw1_acc
        self.dEdw2_acc = eta * self.dEdw2 + self.momentum * self.dEdw2_acc
        self.w1 = self.w1 - self.dEdw1_acc
        self.w2 = self.w2 - self.dEdw2_acc
        
        self.dEdw1 = np.zeros(self.w1.shape)
        self.dEdw2 = np.zeros(self.w2.shape)
        self.cost = np.append(self.cost,0)
        self.ite+=1
        if(self.ite>=10000):
            self.cost = np.zeros(1)
            self.ite=0

targetN = 200
ty = np.linspace(0,1,targetN)
tx = realFunc(ty)+(np.random.rand(targetN)-0.5)*0.2

plotN = 25
x = np.linspace(0,1,plotN)
y = np.linspace(0,1,plotN)
X,Y = np.meshgrid(x, y)
Z = np.ones(X.shape)

def plot(ite):
    
    pi = np.matrix(np.zeros([nn.Npi,plotN]))
    mu = np.matrix(np.zeros([nn.Nmu,plotN]))    
    for i in range(plotN):
        pi[:,i],mu[:,i] = nn.forward_test2(x[i])
        for j in range(plotN):
            Z[i,j] = nn.forward_test(X[i,j],Y[i,j])        


    plt.subplot(2,2,1)
    plt.cla()
    plt.plot(tx,ty,'o',c='None',mec='green')
    plt.plot(x,pi[0,:].transpose())
    plt.plot(x,pi[1,:].transpose())
    plt.plot(x,pi[2,:].transpose())
    plt.axis('equal')
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.01,1.01])

    plt.subplot(2,2,2)
    plt.cla()
    plt.plot(tx,ty,'o',c='None',mec='green')    
    plt.plot(x,mu[0,:].transpose())
    plt.plot(x,mu[1,:].transpose())
    plt.plot(x,mu[2,:].transpose())
    plt.axis('equal')
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.01,1.01])


    plt.subplot(2,2,3)
    plt.cla()

    plt.axis('equal')
    plt.plot(tx,ty,'o',c='None',mec='green')
    plt.contour(X, Y, Z)
    plt.title('time:' + str(ite))
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.01,1.01])

    
#    plt.subplot(1,2,2)
#    plt.cla()                      # 現在描写されているグラフを消去
#    plt.plot(nn.cost,'-')
    


Nhdn = 5
xn = list(range(targetN))
nn = neuralNetwork_mg(1,Nhdn,3)

fig = plt.figure()
def learn(data):
    sbatch=50
    gamma = 0.98
    momentum = 1
    epoch = 1
    eta = (0.001/np.sqrt(sbatch)) * (gamma ** int(data*epoch/100)) * momentum

    for m in range(epoch):
        random.shuffle(xn)
        for n in range(targetN):
            nn.forward(tx[xn[n]],ty[xn[n]])
            nn.backward(ty[xn[n]])
            if(n%sbatch==0):
                nn.updateWeight(eta)
            
    plot(data)

ani = animation.FuncAnimation(fig, learn, interval=1)


#nn.forward(tx[0],ty[0])
#nn.backward(ty[0])
#random.shuffle(xn)
#for m in range(epoch):
#    for n in xn:
#        nn.forward(tx[n],ty[n])
#        nn.backward(ty[n])
#        nn.updateWeight(eta)
#    if(m%100==0):
#        eta *= gamma
#
#plot()

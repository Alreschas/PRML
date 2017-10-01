#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:22:06 2017

@author: Narifumi
"""

import numpy as np
import gzip
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from scipy import optimize
from lib.mnist import mnist

from lib.pooling import pooling
from lib.convolution import convolution
from lib.neuralNetwork import neuralNetwork

plt.clf()

#訓練データ
dataset_dir = '/Users/Narifumi/Documents/Spyder/data/mnist'
mnst=mnist(dataset_dir)


#学習パラメータ
N_hdn = 15 #隠れ層
featureNum1=2#特徴数
featureNum2=2#特徴数
#eta = 0.005
eta = 0.001

N_conv=5 #畳み込み数
pictNum1=1#入力特徴数
pictS1=28#画像サイズ
poolS=2#プーリングサイズ
N_out = 10   #出力クラス数
pictNum2=pictNum1*featureNum1#入力特徴数
pictS2=int(pictS1/poolS)
pictS3=int(pictS2/poolS)

featureN=10
conv1=convolution(N_conv,pictNum1,featureNum1,pictS1)
pool1=pooling(pictNum1*featureNum1,pictS1,poolSize=poolS)
conv2=convolution(N_conv,pictNum2,featureNum2,pictS2)
pool2=pooling(pictNum2*featureNum2,pictS2,poolSize=poolS)

#1回畳み込み
#nn=neuralNetwork((pictS2**2)*pictNum2,N_hdn,N_out)
##2回畳み込み
nn=neuralNetwork((pictS3**2)*pictNum2*featureNum2,N_hdn,N_out)




for j in range(100):
    print('time:',j)
    for i in range(10):
        
           
        #forwardprop
        conv1.inputPict(mnst.data[i].reshape([1,28,28]))
        z_l1=conv1.forward()

        pool1.inputPict(z_l1)
        z_l2=pool1.forward()
        
        conv2.inputPict(z_l2)
        z_l3=conv2.forward()
#        
        pool2.inputPict(z_l3)
        z_l4=pool2.forward()


#        nn.setPict(z_l2)
        nn.setPict(z_l4)
        z_l5=nn.forward()
 
        #目標画像の入力
        nn.setTarget(mnist.num2cls(mnst.label[i]))

        #backwardprop
        nn.backward()    

        pool2.setPreDelta(nn.deltal1[1:].reshape(z_l4.shape))
        pool2.backward()
#        
        conv2.setPreDelta(pool2.delta.reshape(z_l3.shape))
        conv2.backward()
        
#        pool1.setPreDelta(nn.deltal1[1:].reshape(z_l2.shape))        
        pool1.setPreDelta(conv2.delta.reshape(z_l2.shape))       
        pool1.backward()
        
        conv1.setPreDelta(pool1.delta.reshape(z_l1.shape))        
        conv1.backward()

        
        #重みの更新
        nn.update_weight(eta)
        conv1.updateWeight(eta)
        conv2.updateWeight(eta)
    print(mnst.label[i],mnist.cls2num(z_l5))

#予想
okN=0
dataN=10
for i in range(dataN):
    data=mnst.data[i]
    label=mnst.label[i]
    conv1.inputPict(data.reshape([1,28,28]))
    z_l1=conv1.forward()
    
    pool1.inputPict(z_l1)
    z_l2=pool1.forward()
    
    conv2.inputPict(z_l2)
    z_l3=conv2.forward()
    
    pool2.inputPict(z_l3)
    z_l4=pool2.forward()
    
    nn.setPict(z_l4)
    z_l5=nn.forward()
    
#    if(i==5):
#    for p in range(featureN):
#        plt.subplot(4,featureN,1+2*featureN+p)
#        plt.imshow(z_l2[p], cmap=cm.Blues, interpolation='none')
#    plt.subplot(4,featureN,1+2*featureN+p+1)
#    plt.imshow(mnst.data_test[i].reshape([28,28]), cmap=cm.Blues, interpolation='none')
#    
    
    if(label==mnist.cls2num(z_l5)):
        okN+=1
print(okN/dataN * 100)
"""


#1回畳み込み
#for j in range(100):
#    print('time:',j)
#    for i in range(10):
#   
#        #forwardprop
#        conv1.inputPict(mnst.data[i].reshape([1,28,28]))
#        z_l1=conv1.forward()
#        pool1.inputPict(z_l1)
#        z_l2=pool1.forward()
#        nn.setPict(z_l2)
#        z_l3=nn.forward()
# 
#        #目標画像の入力
#        nn.setTarget(mnist.num2cls(mnst.label[i]))
#
#        #backwardprop
#        nn.backward()    
#        
#        pool1.setPreDelta(nn.deltal1[1:].reshape(z_l2.shape))        
#        pool1.backward()
#        
#        conv1.setPreDelta(pool1.delta.reshape(z_l1.shape))        
#        conv1.backward()
#
#        
#        #重みの更新
#        nn.update_weight(eta)
#        conv1.updateWeight(eta)
#    print(mnst.label[i],mnist.cls2num(z_l3))

for i in range(featureN):
    plt.subplot(4,featureN,1+featureN+i)
    plt.imshow(conv1.w[0,i], cmap=cm.Blues, interpolation='none')

#予想
okN=0
dataN=10
for i in range(dataN):
    
    data=mnst.data[i]
    label=mnst.label[i]
    
    conv1.inputPict(data.reshape([1,28,28]))
    z_l1=conv1.forward()    
    pool1.inputPict(z_l1)    
    z_l2=pool1.forward()
    nn.setPict(z_l2)
    z_l3=nn.forward()


    if(i==5):
        for p in range(featureN):
            plt.subplot(4,featureN,1+2*featureN+p)
            plt.imshow(z_l2[p], cmap=cm.Blues, interpolation='none')
        plt.subplot(4,featureN,1+2*featureN+p+1)
        plt.imshow(data.reshape([28,28]), cmap=cm.Blues, interpolation='none')
        
    
    if(label==mnist.cls2num(z_l3)):
        okN+=1
print(okN/dataN * 100)
"""


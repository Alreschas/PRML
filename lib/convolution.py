#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:37:53 2017

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
#from mnist import mnist
#from pooling import pooling


class convolution:
    def __init__(self,convSize,featureSizeIn,featureSize,pictSize):
        self.convSize=convSize
        self.featureSizeIn=featureSizeIn
        self.featureSize=featureSize
        self.pictSize=pictSize
        self.pict = np.zeros([featureSizeIn,pictSize+convSize-1,pictSize+convSize-1])#入力画像

        self.w=np.random.randn(featureSizeIn,featureSize,convSize,convSize)*0.001
        
        self.preDelta=np.zeros([featureSizeIn,featureSize,pictSize,pictSize])
        
        self.delta = np.zeros([featureSizeIn,pictSize,pictSize])
        
        self.dFeatureOut = np.zeros([self.featureSizeIn,self.featureSize,self.pictSize,self.pictSize])
      
        self.dEdW = np.zeros([featureSizeIn,featureSize,convSize,convSize])
        
        self.dEda = np.zeros([featureSizeIn,featureSize,pictSize,pictSize])
        
        self.delta = np.zeros([featureSizeIn,pictSize,pictSize])
        
       
    #    ReLu
    def ReLu(self,x):
        return (x*(x>0))
    
    def dReLu(self,x):
        return (x>0)*1
    
    def setPreDelta(self,delta):
        self.preDelta = delta.reshape(self.featureSizeIn,self.featureSize,self.pictSize,self.pictSize)
    
    def inputPict(self,z):
        for n in range(z.shape[0]):
            self.pict[n] = mnist.zero_padding(z[n],int((self.convSize-1)/2))
            
    def forward(self):
        featureOut=np.zeros([self.featureSizeIn,self.featureSize,self.pictSize,self.pictSize])
        for i in range(self.pictSize):
            for j in range(self.pictSize):
                z=self.pict[:,i:i+self.convSize,j:j+self.convSize]
                for n in range(self.featureSize):
                    a = np.multiply(self.w[:,n],z).reshape(self.featureSizeIn,self.convSize**2).sum(1)
                    featureOut[:,n,i,j] = self.ReLu(a)
                    self.dFeatureOut[:,n,i,j] = self.dReLu(a)
        return featureOut.reshape(self.featureSizeIn*self.featureSize,self.pictSize,self.pictSize)
    
    def backward(self):
        self.delta = np.zeros([self.featureSizeIn,self.pictSize,self.pictSize])
        self.dEda=np.multiply(self.preDelta,self.dFeatureOut)
        for m in range(self.featureSizeIn):
            for n in range(self.featureSize):
                a=mnist.zero_padding(self.dEda[m,n],int((self.convSize-1)/2))
                
        for i in range(self.convSize):
            for j in range(self.convSize):
                z = self.pict[:,i:i+self.pictSize,j:j+self.pictSize]
                for n in range(self.featureSize):
                    self.dEdW[:,n,i,j]=np.multiply(self.dEda[:,n],z).reshape(self.featureSizeIn,self.pictSize**2).sum(1)
                
        w2=self.w[:,:,::-1,::-1]
        for p in range(self.pictSize):
            for q in range(self.pictSize):
                self.delta[:,p,q] += np.multiply(w2,a[p:p+self.convSize,q:q+self.convSize]).reshape(self.featureSizeIn,self.featureSize*self.convSize**2).sum(1)

    def updateWeight(self,eta):
        self.w = self.w - eta * self.dEdW
        self.dEdW = np.zeros([self.featureSizeIn,self.featureSize,self.convSize,self.convSize])
        
if __name__ == '__main__':
    conv = convolution(3,2,3,4)
    
    pict = np.arange(2*4*4).reshape([2,4,4])
    pict[0,1,0]=10
    pict[0,0,3]=10
    print(pict)
    print()
    
    conv.w = np.arange(2*3*3*3).reshape([2,3,3,3])
    print(conv.w)
    print()
    conv.inputPict(pict)
    conv.forward()
    
    conv.backward()
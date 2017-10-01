#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:48:06 2017

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


class neuralNetwork:
    def __init__(self,inSize,hdnSize,outSize):
        self.inSize=inSize
        self.hdnSize=hdnSize
        self.outSize=outSize
        
        self.zl1=np.zeros([inSize+1,1])
        self.zl2=np.zeros([hdnSize+1,1])
        self.zl3=np.zeros([outSize,1])
        
        self.wl1=np.random.randn(inSize+1,hdnSize)*0.001
        self.wl2=np.random.randn(hdnSize+1,outSize)*0.001
        
        self.deltal1=np.zeros([inSize+1,1])
        self.deltal2=np.zeros([hdnSize+1,1])
        self.deltal3=np.zeros([outSize,1])
        self.yt=np.zeros([outSize,1])
        
        self.dEl1=np.zeros([inSize+1,hdnSize])
        self.dEl2=np.zeros([hdnSize+1,outSize])
        

    def setPict(self,pict):
        for n in range(pict.shape[0]):
            size=pict[n].shape[0]*pict[n].shape[1]
            self.zl1[1+size*n:1+size*n+size] = pict[n].reshape([size,1])
            
            
    def getDelta1(self,size,featureN):
        size2=size**2
        return self.deltal1[1+size2*featureN:1+size2*featureN+size2].reshape([size,size])
        
    def setTarget(self,yt):
        self.yt=yt
        
        
    def forward(self):
        self.zl1[0] = 1
        self.zl2[0] = 1
        self.zl2[1:,:]=np.tanh(self.wl1.T.dot(self.zl1))
        zl3_tmp=np.exp(self.wl2.T.dot(self.zl2))
        self.zl3=zl3_tmp/zl3_tmp.sum()
        return self.zl3
    
    
    def backward(self):
        self.deltal3 = (self.zl3-self.yt)
        self.deltal2 = np.multiply(1-np.power(self.zl2,2),self.wl2.dot(self.deltal3))
        self.deltal1 = self.wl1.dot(self.deltal2[1:])
        
        self.dEl2 += self.zl2.dot(self.deltal3.T)
        self.dEl1 += self.zl1.dot(self.deltal2[1:].T)
        
        
    def update_weight(self,eta):
        self.wl1 = self.wl1 - eta * self.dEl1 
        self.wl2 = self.wl2 - eta * self.dEl2
        
        self.dEl1=np.zeros(self.dEl1.shape)
        self.dEl2=np.zeros(self.dEl2.shape)
        
        
        
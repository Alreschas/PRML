#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:14:24 2017

@author: Narifumi
"""

import numpy as np
import gzip
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from scipy import optimize


class pooling:
    
    def __init__(self,featureSize,pictSize,poolSize):
        self.featureSize=featureSize
        self.pictSize=pictSize
        self.poolSize=poolSize
        self.selectPixel=np.zeros([featureSize,pictSize,pictSize])#ピクセル選択フラグ
        self.pict = np.zeros([featureSize,pictSize,pictSize])#入力画像
        self.pictSize_aft = int(pictSize/poolSize)
        self.preDelta=np.zeros([featureSize,self.pictSize_aft,self.pictSize_aft])
        self.delta=np.zeros([featureSize,pictSize,pictSize])
        
        self.mapBefToAft=np.zeros([featureSize,self.pictSize_aft,self.pictSize_aft,poolSize,poolSize])
        
        
    def inputPict(self,z):
        self.pict = z
            
    def setPreDelta(self,delta):
        self.preDelta = delta
        
    def getDelta(self,featureN):
        return self.delta[featureN]
        
    def forward(self):
        self.selectPixel=np.zeros([self.featureSize,self.pictSize,self.pictSize])
        ret=np.zeros([self.featureSize,self.pictSize_aft,self.pictSize_aft])
        for i in range(self.pictSize_aft):
            for j in range(self.pictSize_aft):
                _i1 = i*self.poolSize
                _i2 = i*self.poolSize + self.poolSize 
                _j1 = j*self.poolSize
                _j2 = j*self.poolSize + self.poolSize
                ret[:,i,j] = self.pict[:,_i1:_i2,_j1:_j2].max(axis=1).max(axis=1)
                for n in range(self.featureSize):
                    a=self.pict[n,_i1:_i2,_j1:_j2]
                    p,q=np.unravel_index(a.argmax(), a.shape)
                    self.selectPixel[n,i*2+p,j*2+q] = 1
        return ret

    def backward(self):
        for i in range(self.pictSize_aft):
            for j in range(self.pictSize_aft):
                _i1 = i*self.poolSize
                _i2 = i*self.poolSize + self.poolSize 
                _j1 = j*self.poolSize
                _j2 = j*self.poolSize + self.poolSize
                self.delta[:,_i1:_i2,_j1:_j2] = self.selectPixel[:,_i1:_i2,_j1:_j2] * self.preDelta[:,i,j].reshape(self.featureSize,1,1)
                    
if __name__ == '__main__':
    print()
    pl = pooling(3,4,2)
    pict = np.arange(3*4*4).reshape([3,4,4])
    pict[0,1,0]=10
    pict[0,0,3]=10
    
    pict[1,0,1]=50
    pict[1,1,2]=50
    print(pict)
#    print()
    pl.inputPict(pict)
    ret=pl.forward()
#    print(ret)
#    print()
#    
    delta_next=np.arange(3*2*2).reshape([3,2,2])+10
    print(delta_next)
    print()
    print(pl.selectPixel)
    print()
    pl.setPreDelta(delta_next)
    pl.backward()
    print(pl.delta)
#    print("aaa")
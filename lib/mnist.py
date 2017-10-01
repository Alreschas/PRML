#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:47:03 2017

@author: Narifumi
"""

import numpy as np
import gzip
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from scipy import optimize

class mnist(object):
    
    def __init__(self,dataset_dir):
        key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
        }
        
        self.data,self.label = self.load_mnist(key_file['train_img'],key_file['train_label'],dataset_dir)
        self.data_test,self.label_test = self.load_mnist(key_file['test_img'],key_file['test_label'],dataset_dir)

    def load_mnist(self,img_fname,label_fname,dataset_dir):
        img_size=28*28
        file_path_img = dataset_dir + '/' + img_fname
        file_path_label = dataset_dir + '/' + label_fname
        with gzip.open(file_path_img, 'rb') as f:
            img = np.frombuffer(f.read(), np.uint8, offset=16)
        with gzip.open(file_path_label, 'rb') as f:
            label = np.frombuffer(f.read(), np.uint8, offset=8)
        return img.reshape(-1, img_size),label

    
    def num2cls(y):
        cls=np.zeros([10,1])
        cls[y]=1
        return cls
    def cls2num(cls):
        y = np.argmax(cls)
        return y
    
    def zero_padding(data,expand):
        ret=np.zeros([data.shape[0]+expand*2,data.shape[1]+expand*2])
        ret[expand:expand+data.shape[0],expand:expand+data.shape[1]] = data
        return ret

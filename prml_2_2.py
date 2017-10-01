#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:42:04 2016

@author: Narifumi
"""

#多変量ガウス分布

import numpy as np
import matplotlib.pyplot as plt

def multiGaussian(X,Y,mux,muy,S):
    D=X.shape[0]
    gaussian=np.zeros([D,D])
    mu=np.array([[mux],[muy]])
    for i in range(D):
        for j in range(D):
            x=np.array([[X[i][j]],[Y[i][j]]])
            del2 = np.transpose(x-mu)*np.linalg.inv(S)*(x-mu)
#            /((np.sqrt(np.pi))**2 * np.sqrt(np.linalg.det(S)))
            gaussian[i][j] = np.exp(-del2/2) - np.exp(-0.5)
    return gaussian


S=np.matrix([[1.0,0.8],[0.8,1.0]])

eigval,eigvect=np.linalg.eigh(S)

mux=1.5
muy=1.5

x = np.arange(0,3,0.1)
y = np.arange(0,3,0.1)
X,Y = np.meshgrid(x, y)

g=multiGaussian(X,Y,mux,muy,S)

plt.contour(X, Y,g,levels=[0])
plt.quiver(mux,muy,eigvect[0,0]*np.sqrt(eigval[0]),eigvect[0,1]*np.sqrt(eigval[0]),angles='xy',scale_units='xy',scale=1)
plt.quiver(mux,muy,eigvect[1,0]*np.sqrt(eigval[1]),eigvect[1,1]*np.sqrt(eigval[1]),angles='xy',scale_units='xy',scale=1)
plt.axis('equal')
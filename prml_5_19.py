#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:45:26 2017

@author: Narifumi
"""

import numpy as np
import matplotlib.pyplot as plt
import random

plt.clf()


def realFunc(x):
    ret = x + 0.3 * np.sin(2 * np.pi * x)
    return ret


class neuralNetwork:
    def __init__(self, Nin, Nhdn, Nout):
        self.Nin = Nin
        self.Nhdn = Nhdn
        self.Nout = Nout
        self.w1 = np.random.randn(Nin + 1, Nhdn) * 1
        self.w2 = np.random.randn(Nhdn + 1, Nout) * 1

        self.momentum = 0.9

        self.dEdw1_acc = np.zeros(self.w1.shape)
        self.dEdw2_acc = np.zeros(self.w2.shape)

    def forward(self, x):
        self.z1 = np.append(1, x).reshape(self.Nin + 1, 1)

        a2 = np.dot(self.w1.T, self.z1)
        self.z2 = np.append(1, np.tanh(a2)).reshape(self.Nhdn + 1, 1)

        a3 = np.dot(self.w2.T, self.z2)
        self.z3 = a3

        return self.z3

    def backward(self, yt):
        delta3 = self.z3 - yt
        self.dEdw2 = np.dot(self.z2, delta3.T)

        delta2 = (1 - self.z2**2) * np.dot(self.w2, delta3)
        self.dEdw1 = np.dot(self.z1, delta2[1:, :].T)

    def updateWeight(self, eta):
        self.dEdw1_acc = eta * self.dEdw1 + self.momentum * self.dEdw1_acc
        self.dEdw2_acc = eta * self.dEdw2 + self.momentum * self.dEdw2_acc
        self.w1 = self.w1 - self.dEdw1_acc
        self.w2 = self.w2 - self.dEdw2_acc


targetN = 200
tx1 = np.linspace(0, 1, targetN)
ty1 = realFunc(tx1) + (np.random.rand(targetN) - 0.5) * 0.2

plotN = 1000
x = np.linspace(0, 1, targetN)
y = np.zeros(x.shape)

epoch1 = 500
epoch2 = 2000
eta = 0.005


Nhdn = 5

tx2 = ty1
ty2 = tx1
xn = list(range(targetN))

random.shuffle(xn)
#---------forwardProblem---------#
nn1 = neuralNetwork(1, Nhdn, 1)
for m in range(epoch1):
    for n in xn:
        nn1.forward(tx1[n])
        nn1.backward(ty1[n])
        nn1.updateWeight(eta)

for n in range(x.shape[0]):
    y[n] = nn1.forward(x[n])

plt.subplot(1, 2, 1)
plt.axis('equal')
plt.xlim([-0.2, 1.2])

plt.plot(tx1, ty1, 'o', c='None', mec='green')
plt.plot(x, y, c='r')

#---------inverseProblem---------#
nn2 = neuralNetwork(1, Nhdn, 1)
for m in range(epoch2):
    for n in xn:
        nn2.forward(tx2[n])
        nn2.backward(ty2[n])
        nn2.updateWeight(eta)


for n in range(x.shape[0]):
    y[n] = nn2.forward(x[n])

plt.subplot(1, 2, 2)
plt.axis('equal')
plt.xlim([-0.2, 1.2])
plt.plot(tx2, ty2, 'o', c='None', mec='green')
plt.plot(x, y, c='r')
plt.show()

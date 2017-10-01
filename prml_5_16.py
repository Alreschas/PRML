#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:27:05 2017

@author: Narifumi
"""

import numpy as np
import gzip
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


def rotPict(img, theta):
    # 2次元回転
    c = [14, 14]
    imgret = np.zeros([28, 28])
    rmat = np.matrix((
        (np.cos(theta), -np.sin(theta)),
        (np.sin(theta),  np.cos(theta))
    ))
    for i in range(28):
        for j in range(28):
            x = np.matrix([i - 14, j - 14]).T
            xold = rmat.dot(x) + 14
            if((xold[0] > 0 and xold[1] > 0) and (xold[0] < 28 and xold[1] < 28)):
                imgret[i, j] = img[int(xold[0]), int(xold[1])]
    return imgret


def load_mnist(img_fname, label_fname, dataset_dir):
    img_size = 28 * 28
    file_path_img = dataset_dir + '/' + img_fname
    file_path_label = dataset_dir + '/' + label_fname
    with gzip.open(file_path_img, 'rb') as f:
        img = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(file_path_label, 'rb') as f:
        label = np.frombuffer(f.read(), np.uint8, offset=8)

    return img.reshape(-1, img_size), label


key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = 'data/mnist'
data, label = load_mnist(key_file['train_img'], key_file['train_label'], dataset_dir)

idx = np.random.randint(60000)

plt.subplot(2, 2, 1)
img_org = data[idx].reshape(28, 28)
plt.imshow(img_org, cmap=cm.Blues, interpolation='none')
plt.title(label[idx], color='red')

gsi = 10 * np.pi / 180
eps = 15 * np.pi / 180

# 接ベクトル
plt.subplot(2, 2, 2)
img_p_gsi = rotPict(img_org, gsi)
tau = (img_p_gsi - img_org) / gsi
plt.imshow(tau + 128, cmap=cm.BrBG, interpolation='none')

# 接ベクトルからの寄与を付加
plt.subplot(2, 2, 3)
img3 = img_org + tau * eps
plt.imshow(img3 + np.min(img3), cmap=cm.Blues, interpolation='none')

# ε回転した真の画像
plt.subplot(2, 2, 4)
cmap = cm.Blues
cmap.set_over('w', alpha=0)
img_rot = rotPict(img_org, eps)
plt.imshow(img_rot, cmap=cmap, interpolation='none')

plt.show()
#img_rot2 = rotPict(img_org,np.pi)
#plt.imshow(img_rot2, cmap=cmap, interpolation='none')

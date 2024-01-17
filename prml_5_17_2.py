#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:03:23 2017

@author: Narifumi
"""

import numpy as np
import gzip
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from scipy import optimize

plt.clf()


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


def aFunc_l1(x):
    ret = np.tanh(x)
    return ret


def aFunc_l2(x):
    ret = x
    return ret


def out_l1(x, w_l1):
    ret = np.matrix(np.append(1, aFunc_l1(w_l1.T.dot(x)))).T
    return ret


def out_l2_regression(x, w_l2):
    ret = np.matrix(aFunc_l2(w_l2.T.dot(x))).T
    return ret
# softmax


def out_l2(x, w_l2):
    ret = np.exp(np.matrix(aFunc_l2(w_l2.T.dot(x))).T)
    ret = ret / ret.sum()
    return ret


def num2cls(y):
    cls = np.zeros([y.shape[0], 10])
    for i in range(y.shape[0]):
        cls[i, y[i]] = 1
    return cls


def cls2num(cls):
    y = np.argmax(cls)
    return y

# 回帰問題用のコスト関数(二乗和誤差関数)


def J_regression(param, *args):
    """最小化を目指すコスト関数を返す"""
    w = param
    wl1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
    wl2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)
    # パラメータ以外のデータはargsを通して渡す
    xt, yt = args
    N = xt.shape[0]
    E = 0
    for n in range(N):
        z_l0 = np.matrix(np.append(1, xt[n])).T
        z_l1 = out_l1(z_l0, wl1)
        z_l2 = out_l2(z_l1, wl2)
        y = z_l2[0, 0]
        E += (y - yt[n])**2
    return E / 2

# クラス分類問題用のコスト関数(交差エントロピー誤差関数)


def J_classification(param, *args):
    """最小化を目指すコスト関数を返す"""
    w = param
    wl1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
    wl2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)
    # パラメータ以外のデータはargsを通して渡す
    xt, yt = args
    N = xt.shape[0]
    E = 0
    for n in range(N):
        z_l0 = np.matrix(np.append(1, xt[n])).T
        z_l1 = out_l1(z_l0, wl1)
        z_l2 = out_l2(z_l1, wl2)
        y = z_l2
        E += -(np.multiply(yt[n], np.log(y))).sum()
    #np.save('test.npy', w)
    print(E)
    return E

# 勾配情報(回帰,クラス分類共通)


def gradient(param, *args):
    """コスト関数の偏微分を返す
    各パラメータで偏微分した関数リストを返す"""
    w = param
    wl1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
    wl2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)

    dE1 = np.zeros(wl1.shape)
    dE2 = np.zeros(wl2.shape)

    xt, yt = args
    N = xt.shape[0]

    for n in range(N):
        #        #forward prop
        z_l0 = np.matrix(np.append(1, xt[n])).T
        z_l1 = out_l1(z_l0, wl1)
        z_l2 = out_l2(z_l1, wl2)
        y = z_l2

        delta_l2 = np.matrix(y - yt[n]).T
        delta_l1 = np.multiply(1 - np.power(z_l1, 2), wl2.dot(delta_l2))

        dE_l2 = z_l1.dot(delta_l2.T)
        dE_l1 = z_l0.dot(delta_l1[1:, 0].T)

        dE2 += dE_l2
        dE1 += dE_l1
    dE = np.append(dE1.reshape([(N_in + 1) * N_hdn, 1]), dE2.reshape([(N_hdn + 1) * N_out, 1]))
    return dE


def test(w):
    okn = 0
    testN = 100
    w_l1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
    w_l2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)

    for i in range(testN):
        idx = i + 10000  # np.random.randint(60000)
        z0 = np.append(1, data[idx])
        z1 = out_l1(z0, w_l1)
        z2 = out_l2(z1, w_l2)
        if(label[idx] - cls2num(z2) == 0):
            okn += 1
    p = okn / testN * 100
    return p


# 訓練データ
dataset_dir = 'data/mnist'
data, label = load_mnist(key_file['train_img'], key_file['train_label'], dataset_dir)
data_test, label_test = load_mnist(key_file['test_img'], key_file['test_label'], dataset_dir)

N = 10


# 学習パラメータ
N_in = 28 * 28  # 画像サイズ
N_out = 10  # 出力クラス数
N_hdn = 64  # 隠れ層
w=np.random.randn((N_in+1)*N_hdn+(N_hdn+1)*N_out)*np.sqrt(10)
#w = np.load('test.npy')

idx = 0

N_td = 6000


# print(data[idx:idx+N_td].shape)
for i in range(0):
    # 訓練データ(ミニバッチ)の作成
    idx = i * N_td
    x_t = data[idx:idx + N_td]
    y_t = num2cls(label[idx:idx + N_td])

    # 学習前の交差エントロピー誤差関数
    J_classification(w, *(x_t, y_t))

    # 確率的勾配降下法
#    eta=0.001
#    for k in range(100):
#        w = w - eta * gradient(w,*(x_t,y_t))
#        print(k)

    # 共役勾配法
    w = optimize.fmin_cg(J_classification, w, fprime=gradient, args=(x_t, y_t), gtol=1)

    # 学習後の交差エントロピー誤差関数
    J_classification(w, *(x_t, y_t))


# チェック
print("TEST")
w_l1 = w[:(N_in + 1) * N_hdn].reshape(N_in + 1, N_hdn)
w_l2 = w[(N_in + 1) * N_hdn:].reshape(N_hdn + 1, N_out)
okn = 0
testN = 5000
for i in range(testN):
    #    idx=np.random.randint(60000)
    idx = i
    z0 = np.append(1, data_test[idx])
    z1 = out_l1(z0, w_l1)
    z2 = out_l2(z1, w_l2)
    if(i < 50):
        plt.subplot(5, 10, i + 1)
        img_org = data_test[idx].reshape(28, 28)
        plt.imshow(img_org, cmap=cm.Blues, interpolation='none')
        plt.title(cls2num(z2), color='red')
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
#    print(label[idx],cls2num(z2))
    if(label_test[idx] == cls2num(z2)):
        okn += 1
print(okn / testN * 100, "%")

plt.show()

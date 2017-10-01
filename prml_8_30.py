#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# エネルギーの差分から、尤もらしいxjの値を返す
def update_xj(x, y, j, beta, eta, h):
    rows = x.shape[0]
    cols = x.shape[1]

    # 隣接ノード xjから見て左右・上下
    neiNode = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

    # xjが+1の場合と-1の場合のエネルギーの差分を計算
    engDiff = -2 * eta * y[j[0], j[1]] + 2 * h
    for n in neiNode:
        i = j - n
        if((0 <= i[0] and i[0] < rows) and (0 <= i[1] and i[1] < cols)):
            engDiff += -2 * beta * x[i[0], i[1]]

    # エネルギーを減らす方向にxの値を更新
    ret = 1
    if(engDiff >= 0):
        ret = -1

    return ret


# 画像の読み込み
# 真の画像
orgImg = np.loadtxt('data/figure/figure8.30a.csv', delimiter=",", dtype=int)

# 観測画像
y = np.loadtxt('data/figure/figure8.30b.csv', delimiter=",", dtype=int)


# パラメータ
beta = 1.0
eta = 2.1
h = 0

# ノイズ除去処理反復回数
iteMax = 2

# ICMによるノイズ除去
x = np.array(y)
for ite in range(iteMax):
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            j = np.array([row, col])
            x[j[0], j[1]] = update_xj(x, y, j, beta, eta, h)


# 画像の表示
plt.subplot(2, 2, 1)
plt.tick_params(left='off', right='off', top='off', bottom='off', labelleft='off', labelbottom='off')
plt.imshow(orgImg, cmap='gray')

plt.subplot(2, 2, 2)
plt.tick_params(left='off', right='off', top='off', bottom='off', labelleft='off', labelbottom='off')
plt.imshow(y, cmap='gray')

plt.subplot(2, 2, 3)
plt.tick_params(left='off', right='off', top='off', bottom='off', labelleft='off', labelbottom='off')
plt.imshow(x, cmap='gray')

#import scipy as sp
#sp.misc.imsave('data/figure/result.png', x)

plt.show()

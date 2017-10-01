#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# プロット用データ
plotS = 100
X = np.linspace(-0.5, 1.5, plotS)

# 訓練データ
data = np.loadtxt('data/curvefitting.txt', comments='#', delimiter=' ')
x = data[:, 0]
t = data[:, 1]
#dataS = 10
#x = np.linspace(0,1,dataS)
#t = np.sin(2 * np.pi * x) + np.random.randn(dataS) * 0.3 + 5


def _phi(x, mu):
    theta = 20
    ret = np.append(np.exp(-theta * (x - mu)**2), 1)
#    ret = np.exp(-theta * (x-mu)**2)
    return ret


def _Phi(x):
    N = x.shape[0]
    ret = _phi(x[0], x)
    for n in range(1, N):
        ret = np.vstack([ret, _phi(x[n], x)])
    return ret


def norm2(x):
    ret = np.dot(x, x)
    return ret


# 訓練データのサイズ
N = x.shape[0]

# 特徴ベクトルの次元
M = N + 1

alpha = np.ones([M])
beta = 1
Phi = _Phi(x)

# αの上限
sup_alpha = 1e10

# 学習
for i in range(1000):
    invSig = np.diag(alpha) + beta * np.dot(Phi.T, Phi)
    Sig = np.linalg.inv(invSig)
    m = beta * Sig.dot(Phi.T).dot(t)
    gamma = 1 - alpha * np.diag(Sig)
    alpha = gamma / (m**2)
    beta = (N - gamma.sum()) / norm2(t - Phi.dot(m))

    # 値が発散しないように制限
    for j in range(M):
        if(alpha[j] >= sup_alpha):
            alpha[j] = sup_alpha


# 全てのデータを利用した場合
plt.subplot(2, 1, 1)

# 全てのxで予測値を作成
Y = np.zeros([plotS])
sig = np.zeros([plotS])
for i in range(plotS):
    phi = _phi(X[i], x)
    Y[i] = np.dot(m, phi)
    sig[i] = np.sqrt(1 / beta + phi.dot(Sig).dot(phi))
plt.plot(x, t, 'o', c='none', markeredgecolor='g', markeredgewidth=1)
plt.plot(X, Y, 'r')
plt.fill_between(X, Y + sig, Y - sig, color='pink', alpha=0.5)


# 関連ベクトルのみを利用した場合
plt.subplot(2, 1, 2)

# 関連ベクトル以外削除
x_sv = np.array(x)
m_sv = np.array(m)
t_sv = np.array(t)
Phi_sv = np.array(Phi)
alpha_sv = np.array(alpha)
for i in range(M - 2, -1, -1):
    if(alpha[i] >= sup_alpha):
        x_sv = np.delete(x_sv, i)
        t_sv = np.delete(t_sv, i)
        m_sv = np.delete(m_sv, i)
        alpha_sv = np.delete(alpha_sv, i)
        Phi_sv = np.delete(Phi_sv, i, axis=1)

# 全てのxで予測値を作成
invSig_sv = np.diag(alpha_sv) + beta * np.dot(Phi_sv.T, Phi_sv)
Sig_sv = np.linalg.inv(invSig_sv)
Y_sv = np.zeros([plotS])
sig_sv = np.zeros([plotS])
for i in range(plotS):
    phi_sv = _phi(X[i], x_sv)
    Y_sv[i] = np.dot(m_sv, phi_sv)
    sig_sv[i] = np.sqrt(1 / beta + phi_sv.dot(Sig_sv).dot(phi_sv))

# プロット
plt.plot(x, t, 'o', c='none', markeredgecolor='g', markeredgewidth=1)
plt.plot(x_sv, t_sv, 'o', c='none', ms=12, markeredgecolor='blue', markeredgewidth=1)
plt.plot(X, Y_sv, 'r', alpha=0.7)
plt.fill_between(X, Y + sig_sv, Y - sig_sv, color='pink', alpha=0.5)

plt.show()

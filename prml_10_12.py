import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def sigmoid_lowerLimit(x, xi):
    lam = np.tanh(xi / 2) / (4 * xi)
    y = sigmoid(xi) * np.exp((x - xi) / 2 - lam * (x**2 - xi**2))
    return y


def sigmoid_upperLimit(x, eta):
    g = - (1 - eta) * np.log(1 - eta) - eta * np.log(eta)
    y = np.exp(eta * x - g)
    return y


def sigmoid_upperLimit2(x, xi):
    eta = 1 / (1 + np.exp(xi))
    g = eta * xi - np.log(sigmoid(xi))
    y = np.exp(eta * x - g)
    return y


x = np.linspace(-6, 6, 100)

plt.subplot(1, 2, 1)
plt.ylim([0, 1])
plt.plot(x, sigmoid(x), 'k-')

# シグモイド
# シグモイドの上限
eta1 = 0.2
eta2 = 0.7
plt.plot(x, sigmoid_upperLimit(x, eta1), 'k--')
plt.plot(x, sigmoid_upperLimit(x, eta2), 'k--')

plt.subplot(1, 2, 2)
plt.ylim([0, 1])
plt.plot(x, sigmoid(x), 'k-')

xi = 2.5
# シグモイドの下限
plt.plot(x, sigmoid_lowerLimit(x, xi), 'k--')
# シグモイドの上限
plt.plot(x, sigmoid_upperLimit2(x, xi), 'k--')

plt.vlines(xi, 0, sigmoid(xi), linestyles='dotted')
plt.vlines(-xi, 0, sigmoid(-xi), linestyles='dotted')


plt.show()

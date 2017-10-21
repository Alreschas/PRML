import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-20, 20, 1000)


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def lam(x):
    y = (sigmoid(x) - 0.5) / (2 * x)
    return y


def lamd(x):
    y = - (sigmoid(x) - 0.5) / (2 * (x**2)) + sigmoid(x) * (1 - sigmoid(x)) / (2 * x)
    return y


plt.subplot(2, 1, 1)
y = lam(x)
plt.plot(x, y)

plt.subplot(2, 1, 2)
y = lamd(x)
plt.plot(x, y)


plt.show()

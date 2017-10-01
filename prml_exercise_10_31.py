import numpy as np
import matplotlib.pyplot as plt


def f(x):
    y = -np.log(np.exp(x / 2) + np.exp(-x / 2))
    return y


x = np.linspace(-50, 50, 100)

# xについてのグラフ
plt.subplot(2, 1, 1)
plt.plot(x, f(x))

# x^2についてのグラフ
plt.subplot(2, 1, 2)
plt.plot(x**2, f(np.sqrt(x**2)))

plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 50).reshape(-1, 1)
w = np.array([[-1, 0, 1]])
b = np.array([-1, 0, -2])
z = x @ w + b
y = np.max(z, axis=1)

plt.plot(x, z[:, 0])
plt.plot(x, z[:, 1])
plt.plot(x, z[:, 2])
plt.plot(x, y, lw=3)
plt.grid(True)
plt.gca().set_aspect("equal")
plt.show()

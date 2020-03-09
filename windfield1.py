import matplotlib.pyplot as plt
import numpy as np

x1, y1 = np.meshgrid(np.arange(-2, 2, .5), np.arange(-2, 2, .5))
# z = x*np.exp(-x**2 - y**2)
# v, u = np.gradient(z, .2, .2)
vx = x1 - y1
vy = y1 + x1

fig, ax = plt.subplots()

q = ax.quiver(x1, y1, vx, vy)

plt.show()
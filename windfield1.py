import matplotlib.pyplot as plt
import numpy as np

x,y = np.meshgrid(np.arange(-2, 2, .25), np.arange(-2, 2, .25))
# z = x*np.exp(-x**2 - y**2)
# v, u = np.gradient(z, .2, .2)
v = x - y
u = y + x

fig, ax = plt.subplots()

q = ax.quiver(x, y, u, v)
plt.show()
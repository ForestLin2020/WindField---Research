import matplotlib.pyplot as plt
import numpy as np

x, y = np.meshgrid(np.arange(-2, 3, 1), np.arange(-2, 3, 1))

# original point is x and y when vx=0 and vy =0
vx = x-y
vy = y+x

fig, ax = plt.subplots()

# ax.quiver(x, y, vx, vy,units='xy', scale=1)
ax.quiver(x, y, vx, vy)
plt.axis([-5, 5, -5, 5])
plt.grid()
# set the grid (aspect as rectangle('auto' or num) or square('equal'))
ax.set_aspect('equal')

# draw the end points of vectors
x_end = x + vx/5
y_end = y + vy/5
plt.scatter(x_end, y_end, s=10)

# draw a circle for a drone
path_circle = plt.Circle((0, 0), 3, color='r', fill=False)
ax.add_artist(path_circle)

plt.show()
plt.close()
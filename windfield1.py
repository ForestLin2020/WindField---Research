import matplotlib.pyplot as plt
import numpy as np

x, y = np.meshgrid(np.arange(-2, 3, 1), np.arange(-2, 3, 1))
# z = x*np.exp(-x**2 - y**2)
# v, u = np.gradient(z, .2, .2)
# original point is x and y when vx=0 and vy =0
vx = x-y
vy = y+x

print('point-x:',x)
print('point-y:',y)
print('point-y+x:',x+y)
print('point-(y+x)/10:',(x+y)/10)

print('vx',vx)
print('vy',vy)

fig, ax = plt.subplots()

# ax.quiver(x, y, vx, vy,units='xy', scale=1)
ax.quiver(x, y, vx, vy)
plt.axis([-5, 5, -5, 5])


plt.grid()
# set the grid (aspect as rectangle('auto' or num) or square('equal'))
ax.set_aspect('equal')

plt.show()
plt.close()
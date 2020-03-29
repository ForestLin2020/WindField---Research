import matplotlib.pyplot as plt
import numpy as np





n= 90
x = np.linspace(-6, 6, n).reshape(-1, 1)    # test points
print('x = ',x)
y = -np.linspace(-6, 6, n).reshape(-1, 1)    # test points
# y = np.linspace(-3, 3, n).reshape(-1, 1)
# print('ytrain = ',ytrain)

# x, y = np.meshgrid(np.arange(-2, 3, 1), np.arange(-2, 3, 1))
# z = x*np.exp(-x**2 - y**2)
# v, u = np.gradient(z, .2, .2)
# original point is x and y when vx=0 and vy =0
vx = x-y
vy = y+x

x_number_list = x + vx/5
y_number_list = y + vy/5

# Draw point based on above x, y axis values.
plt.scatter(x_number_list, y_number_list, s=10)
plt.grid()
# Set chart title.
plt.title("Extract Number Root ")

# Set x, y label text.
plt.xlabel("Number")
plt.ylabel("Extract Root of Number")

plt.show()
plt.close()




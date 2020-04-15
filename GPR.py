import numpy as np
import matplotlib.pyplot as pl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# Test data
n = 100  # number of test points

# np.linspace(-6, 6, n)     # -6 ~ 6 separate in to n
# reshape(-1, 1)            # reshape(-1,1) (-1: unknown/don't care, 1: column=1)
Xtest, Ytest = np.meshgrid(np.linspace(-6, 6, n).reshape(-1, 1), np.linspace(-6, 6, n).reshape(-1, 1))
print('Xtest',Xtest)
# Test_potints = np.linspace(-6, 6, n).reshape(-1, 1)
# Test_potints = np.linspace(-6, 6, n).reshape(-1, 1)
# print('Test_potints',Test_potints)





# Define the kernel function: cholesky => Total sum = L * L.T
# Array 0D >> np.mutiply or *
# Array 1D >> np.dot
# Array 2D >> np.matmul or @

# kernel_2
sigma_f = 1
lamb = 0.1

# kernel width parameter = ell**2
param = 0.4


uav_x_list = []
uav_y_list = []

x_path_train_points_list = []
y_path_train_points_list = []

def kernel_2(a, b, c, d):
    a_b_2 = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    c_d_2 = np.sum(c**2,1).reshape(-1,1) + np.sum(d**2,1) - 2*np.dot(c, d.T)
    e = np.sqrt(a_b_2 + c_d_2)
    return sigma_f**2 * np.exp(-lamb*np.abs(e))

# def kernel_2(a, b, c, d):
#     a_b_2 = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#     c_d_2 = np.sum(c**2,1).reshape(-1,1) + np.sum(d**2,1) - 2*np.dot(c, d.T)
#     e = np.sqrt(a_b_2 + c_d_2)
#     return sigma_f**2 * np.exp(-lamb*np.abs(e))

def kernel(a, b):
    # np.sum(a,axis=1), axis = 1, every [1,1],[2,2] sum themself
    # np.sum(a,axis=0), axis = 0, every [1,2],[1,2] sum the same position elements
    # sum will make 2D become 1D

    #            2D                    +       1D
    # np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1)
    # >> become a Sigma matrix [ sigma11, sigma12 ,sigma13, ... ]
    #                          [ sigma21, sigma22, sigma32, ... ]

    # notice the equation is 2D + 1D

    a_b_2 = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return sigma_f**2 * np.exp(-.5 * (1/param) * a_b_2)

def init():
    ax.add_patch(uav)
    return uav,

def animate(i):
    # i: 0~359
    uav_x = 0 + 4 * np.sin(np.radians(i))
    uav_y = 0 + 4 * np.cos(np.radians(i))
    uav.center = (uav_x, uav_y)
    # print('uav.center', uav.center)
    if len(uav_x_list) < 360:
        uav_x_list.append(uav_x)
        uav_y_list.append(uav_y)
        # print('uav_x_list',len(uav_x_list))
        # print('uav_y_list',len(uav_y_list))

    return uav,

# range_train_points = 8
T = 10
# Training points in grid
# X = np.arange(-6, 6, n)
# Y = np.arange(-6, 6, n)
# x, y = np.meshgrid(X, Y)


# x, y = np.meshgrid(np.arange(-R/2, R/2, 1), np.arange(-R/2, R/2, 1))
xtrain, ytrain = np.meshgrid(np.linspace(-5, 5, T).reshape(-1, 1), np.linspace(-5, 5, T).reshape(-1, 1))



vx = (xtrain - ytrain)  # X train = x + vx
vy = (ytrain + xtrain)  # f(X train) = y + vy

# print('trainx',x.shape)
# print('trainy',y.shape)
# print('vx',vx)
# print('uav_x_list',len(uav_x_list))


# then plus path now
# first assume circle is our path
path_circle = pl.Circle((0, 0), 4, color='r', fill=False)
uav = pl.Circle((1, -1), 0.3, fc='r')
fig, ax = pl.subplots()
ax.quiver(xtrain, ytrain, vx, vy)
ax.add_artist(path_circle)
# pl.axis([-15, 15, -15, 15])
pl.grid()
# set the grid (aspect as rectangle('auto' or num) or square('equal'))
ax.set_aspect('equal')
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

pl.show()


path_train_points_number = 360 / T
# print('path_train_points_number',path_train_points_number)
for i in range(len(uav_x_list)):
    if i % path_train_points_number == 0:
        x_path_train_points_list.append(uav_x_list[i])
        y_path_train_points_list.append(uav_y_list[i])
        # print('x_path_train_points', x_path_train_points_list)
        # print('y_path_train_points', y_path_train_points_list)
        # print('x_path_train_points', len(x_path_train_points_list))
        # print('y_path_train_points', len(y_path_train_points_list))


x_path_train_points_np = np.array(x_path_train_points_list).reshape(-1,1)
y_path_train_points_np = np.array(y_path_train_points_list).reshape(-1,1)


# K_ss = kernel(Xtest, Xtest)  # Kernel at test points - Radial basis function kernel
K_ss = kernel_2(Xtest, Xtest,
                Ytest, Ytest)  # Kernel at test points - Radial basis function kernel
# Lss is the square root of Kss
L_ss = np.linalg.cholesky(K_ss + 1e-6*np.eye(n))
# print('L_ss',L_ss.shape)
# print('L_ss',L_ss)
# f_prior = np.dot(L_ss, np.random.normal(size=(n,3)))
f_prior = L_ss @ np.random.normal(size=(n,3))


# # Now let's plot the 3 sampled functions.
# pl.figure(2)
# pl.plot(Test_potints, f_prior)
# pl.axis([-6, 6, -3, 3])
# pl.title('Three samples from the GP prior')
# pl.show()


# xmu_3d = np.ones((T,n))
# ymu_3d = np.ones((T,n))

############ for loop from there: y.shape[1] means run every row #########3
# for i in range(T):
#
#     print('xtrain[i]',xtrain[i].reshape(-1,1))
#     print('Xtest[i]',Xtest[i].reshape(-1,1))
#     print('xtrain[i]',xtrain[i].reshape(-1,1).shape)
#     print('Xtest[i]',Xtest[i].reshape(-1,1).shape)

# Noiseless training data "D"
# Training points: knew points which is using to predict unknown points
# Train_points = x[0].reshape(-1,1)

# Apply the kernel function to our training points
# K(x, x') >>
# x: start_point ~ end_point
# x': start_point ~ end_point

# K = kernel(xtrain[i].reshape(-1,1), xtrain[i].reshape(-1,1))
K = kernel_2(xtrain, xtrain, ytrain, ytrain)
# K = kernel_2(xtrain[i].reshape(-1,1), xtrain[i].reshape(-1,1),
#              ytrain[i].reshape(-1,1), ytrain[i].reshape(-1,1))  # Kernel at test points - Radial basis function kernel
L = np.linalg.cholesky(K + 0.04*np.eye(len(xtrain)))  # why need to + 0.00005?
print('K.shape(10x10)', K.shape)

# Compute the mean at our test points.
# K_s = kernel(xtrain[i].reshape(-1,1), Xtest[i].reshape(-1,1))
K_s = kernel_2(xtrain, Xtest,
               ytrain, Ytest)
# K_s = kernel_2(xtrain[i].reshape(-1,1), Xtest[i].reshape(-1,1),
#                ytrain[i].reshape(-1,1), Ytest[i].reshape(-1,1))  # Kernel at test points - Radial basis function kernel
Lk = np.linalg.solve(L, K_s)
print('K_s.shape(10x100)', K_s.shape)


# np.linalg.solve(NxN, Nx1)
# L: 5x5 , vector_x or vector_y: 5x1
x_mu = np.dot(Lk.T, np.linalg.solve(L, vx[i].reshape(-1,1))).reshape((n,))
# y_mu = np.dot(Lk.T, np.linalg.solve(L, vy[i].reshape(-1,1))).reshape((n,))
# xmu_3d[i] = x_mu
# ymu_3d[i] = y_mu
print('mu', x_mu.shape)
# print('mu', xmu_3d.shape)

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)

# Draw samples from the posterior at our test points.
# L = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(Lk.T, Lk))

# y_post = y_mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))     # posteriors: The GP function follow the training point
# x_post = x_mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))     # posteriors: The GP function follow the training point


# prediction on wind vector of x
pl.figure(3)
# Training points (X,Y)
pl.plot(xtrain[i],vx[i].reshape(-1,1), 'bo', ms=8, label='Prediction')
# # post line
# pl.plot(Test_potints, x_post)
# draw uncertain area
pl.gca().fill_between(Xtest[i].flat, x_mu-stdv, x_mu+stdv, color="#dddddd")
# draw the mean line(--)
pl.plot(Xtest[i], x_mu, 'r--', lw=2)
# axis range of x and y [x_left, x_right, y_left, y_right]
# pl.axis([-6, 6, -5, 5])
pl.title('Prediction on Vector of x')

# # prediction on wind vector of y
# pl.figure(4)
# # Training points (X,Y)
# pl.plot(x, vy[i].reshape(-1,1), 'bo', ms=8)
# # post line
# pl.plot(Test_potints, y_post)
# # draw uncertain area
# pl.gca().fill_between(Test_potints.flat, y_mu-stdv, y_mu+stdv, color="#dddddd")
# # draw the mean line(--)
# pl.plot(Test_potints, y_mu, 'r--', lw=2)
# print('Test_potints',Test_potints.shape)
# print('y_mu',y_mu.shape)
# # axis range of x and y [x_left, x_right, y_left, y_right]
# # pl.axis([-6, 6, -5, 5])
# pl.title('Prediction on Vector of y')
# pl.show()



# 3D (x)
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

# print('Xtest', Xtest)
print('Xtest', Xtest.shape)
print('x_mu', x_mu)
# print('xmu_3d', xmu_3d.shape)

# Make data.
X = Xtest
Y = Xtest
# X, Y = np.meshgrid(X, Y)
Z = x_mu
# Z = xmu_3d


# print('Xtest',Xtest)
# print('xmu_3d',xmu_3d)
# print('Xtest.shape',Xtest.shape)
# print('xmu_3d.shape',xmu_3d.shape)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-2.01, 2.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#
#
#
#
# # 3D (y)
# fig = pl.figure()
# ax = fig.add_subplot(111, projection='3d')
# Z = ymu_3d
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
# # Customize the z axis.
# # ax.set_zlim(-2.01, 2.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
#
pl.show()

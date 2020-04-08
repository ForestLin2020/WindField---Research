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
Test_potints = np.linspace(-6, 6, n).reshape(-1, 1)




# Define the kernel function: cholesky => Total sum = L * L.T
# Array 0D >> np.mutiply or *
# Array 1D >> np.dot
# Array 2D >> np.matmul or @


def kernel_2(a, b, c, d, lamb, sigma_f):
    a_b_2 = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    c_d_2 = np.sum(c**2,1).reshape(-1,1) + np.sum(d**2,1) - 2*np.dot(c, d.T)
    e = np.sqrt(a_b_2 + c_d_2)
    return sigma_f**2 * np.exp(-lamb*np.abs(e))

def kernel(a, b, param):
    # np.sum(a,axis=1), axis = 1, every [1,1],[2,2] sum themself
    # np.sum(a,axis=0), axis = 0, every [1,2],[1,2] sum the same position elements
    # sum will make 2D become 1D

    #            2D                    +       1D
    # np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1)
    # >> become a Sigma matrix [ sigma11, sigma12 ,sigma13, ... ]
    #                          [ sigma21, sigma22, sigma32, ... ]

    # notice the equation is 2D + 1D

    a_b_2 = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * a_b_2)

def init():
    ax.add_patch(uav)
    return uav,

def animate(i):
    x = 0 + 4 * np.sin(np.radians(i))
    y = 0 + 3 * np.cos(np.radians(i))
    uav.center = (x, y)
    return uav,



# kernel width parameter = ell**2
param = 0.1
K_ss = kernel(Test_potints, Test_potints, param)  # Kernel at test points - Radial basis function kernel




'''
# draw samples from the prior at our test points
# Get cholesky decomposition (square root) of the covariance matrix
K_temp = 0.5 * (K_ss.T + K_ss)
# K_I = np.linalg.eig(K_temp)
print('K_temp_Igan = ', np.linalg.eig(K_temp))
L_ss = np.linalg.cholesky(K_temp)
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L_ss, np.random.normal(size=(n,3)))
'''

# Lss is the square root of Kss
L_ss = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
# print('L_ss',L_ss.shape)
# print('L_ss',L_ss)
# f_prior = np.dot(L_ss, np.random.normal(size=(n,3)))
f_prior = L_ss @ np.random.normal(size=(n,3))


# # Now let's plot the 3 sampled functions.
# pl.figure(1)
# pl.plot(Test_potints, f_prior)
# pl.axis([-6, 6, -3, 3])
# pl.title('Three samples from the GP prior')

# range_train_points = 8
R = 100
# measurepoint points in grid
x, y = np.meshgrid(np.arange(-R/2, R/2, 1), np.arange(-R/2, R/2, 1))
vx = (x - y)/5
vy = (y + x)/5
print('X',x.shape)
print('Y',y.shape)
print('Z',vx.shape)

# then plus path now
# first assume circle is our path
path_circle = pl.Circle((0, 0), 3, color='r', fill=False)
uav = pl.Circle((5, -5), 0.3, fc='r')

fig, ax = pl.subplots()
ax.quiver(x, y, vx, vy)
ax.add_artist(path_circle)
pl.axis([-5, 5, -5, 5])
pl.grid()
# set the grid (aspect as rectangle('auto' or num) or square('equal'))
ax.set_aspect('equal')
animate =
anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=20,
                               blit=True)

pl.show()

xmu_3d = np.ones((R,n))
ymu_3d = np.ones((R,n))

############ for loop from there: y.shape[1] means run every row #########3
for i in range(R):

    # Noiseless training data "D"
    # Training points: knew points which is using to predict unknown points
    Train_points = x[0].reshape(-1,1)

    # Apply the kernel function to our training points
    # K(x, x') >>
    # x: start_point ~ end_point
    # x': start_point ~ end_point
    K = kernel(Train_points, Train_points, param)
    L = np.linalg.cholesky(K + 0.00005*np.eye(len(Train_points)))  # why need to + 0.00005?

    # Compute the mean at our test points.
    K_s = kernel(Train_points, Test_potints, param)
    Lk = np.linalg.solve(L, K_s)

    # np.linalg.solve(NxN, Nx1)
    # L: 5x5 , vector_x or vector_y: 5x1
    x_mu = np.dot(Lk.T, np.linalg.solve(L, vx[i].reshape(-1,1))).reshape((n,))
    y_mu = np.dot(Lk.T, np.linalg.solve(L, vy[i].reshape(-1,1))).reshape((n,))
    xmu_3d[i] = x_mu
    ymu_3d[i] = y_mu



    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)

    # Draw samples from the posterior at our test points.
    L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))

    # y_post = y_mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))     # posteriors: The GP function follow the training point
    # x_post = x_mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))     # posteriors: The GP function follow the training point


    # # prediction on wind vector of x
    # pl.figure(3)
    # # Training points (X,Y)
    # pl.plot(Train_points,vx[i].reshape(-1,1), 'bs', ms=8)
    # # post line
    # # pl.plot(Test_potints, x_post)
    # # draw uncertain area
    # pl.gca().fill_between(Test_potints.flat, x_mu-stdv, x_mu+stdv, color="#dddddd")
    # # draw the mean line(--)
    # pl.plot(Test_potints, x_mu, 'r--', lw=2)
    # # axis range of x and y [x_left, x_right, y_left, y_right]
    # # pl.axis([-6, 6, -5, 5])
    # pl.title('Prediction on Vector of x')
    #
    # # prediction on wind vector of y
    # pl.figure(4)
    # # Training points (X,Y)
    # pl.plot(Train_points, vy[i].reshape(-1,1), 'bs', ms=8)
    # # post line
    # # pl.plot(Test_potints, y_post)
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



# 3D (x,y,)
fig = pl.figure()
ax = fig.add_subplot(111,projection='3d')


# x, y = np.meshgrid(Test_potints,Test_potints)

# Make data.
X = Test_potints
Y = Test_potints
X, Y = np.meshgrid(X, Y)

Z = xmu_3d
# R = np.sqrt(X**2 + Y**2)
# Z = np.cos(R)

print('X',X.shape)
print('Y',Y.shape)
print('Z',Z.shape)
print('xmu_3d',xmu_3d.shape)

# Plot the surface.
surf = ax.plot_surface(X, Y, xmu_3d,cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-2.01, 2.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
pl.show()

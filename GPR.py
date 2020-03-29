import numpy as np
import matplotlib.pyplot as pl


# Test data
n = 100    # number of test points

Xtest_1 = np.linspace(-6, 6, n)     # format:(90,) tuple type
Xtest = Xtest_1.reshape(-1, 1)      # reshape(-1,1) row:unknown colume
print('Xtest_1 = ', Xtest_1)
print('Xtest = ', Xtest)

# Xtest = np.linspace(-6, 6, n).reshape(-1, 1)    # test points

# Define the kernel function: cholesky => Total sum = L * L.T
# Array 0D >> np.mutiply or *
# Array 1D >> np.dot
# Array 2D >> np.matmul or @ 


def kernel(a, b, param):
    # np.sum(a,axis=1), axis = 1, every array sum themself
    # np.sum(a,axis=0), axis = 0, every array sum the same position elements
    # a1 = np.sum(a ** 2, 1).reshape(-1, 1)
    # b1 = np.sum(b ** 2, 1)
    # c1 = 2 * np.dot(a, b.T)
    # sqdist = a1 + b1 - c1
    # print('a = ', a)
    # print('b = ', b)
    # print('param = ', param)
    # print('a1 = ',a1)
    # print('b1 = ',b1)
    # print('c1 = ',c1)

    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    print('sqdist = ',sqdist)
    return np.exp(-.5 * (1/param) * sqdist)



param = 0.1 # kernel width parameter = l**2
K_ss = kernel(Xtest, Xtest, param)  # Kernel at test points - Radial basis function kernel

# print('K_ss.shape = ', K_ss.shape)
# print('K_ss =',K_ss)
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


L_ss = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
f_prior = np.dot(L_ss, np.random.normal(size=(n,3)))


# Now let's plot the 3 sampled functions.
pl.plot(Xtest, f_prior)
pl.axis([-6, 6, -3, 3])
pl.title('Three samples from the GP prior')
pl.show()



# Noiseless training data "D"
# Training points: knew points which is using to predict unknown points
Xtrain = np.array([-2, -1, 0, 1, 2]).reshape(5,1)
# print('Xtrain = ',Xtrain)
ytrain =  np.array([1, 1, 1, 1, 1]).reshape(5,1)
# print('ytrain = ',ytrain)

vector_x = Xtrain-ytrain
vector_y = Xtrain+ytrain

#point + vector
Xtrain = Xtrain + vector_x/5
ytrain = ytrain + vector_y/5


# Apply the kernel function to our training points
# K(x, x') >>
# x: start_point ~ end_point
# x': start_point ~ end_point
K = kernel(Xtrain, Xtrain, param)
print('K= ',K)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))  # why need to + 0.00005?

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)

# Lkk = np.linalg.solve(L,Lk)                   # 1
# mu = np.dot(Lkk.T,  ytrain).reshape((n,))     # 2
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))     # 1+2

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)

# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))     # posteriors: The GP function follow the training point

pl.plot(Xtrain, ytrain, 'bs', ms=8)
pl.plot(Xtest, f_post)
pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.axis([-6, 6, -6, 6])
pl.title('Three samples from the GP posterior')
pl.show()
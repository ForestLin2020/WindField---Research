import numpy as np

a = np.array([[2,2],[4,4],[5,5],[6,6]])
b = np.array([[2,4,5,6]])

print('a',a)
print('a.ndim',a.ndim)
print('a**2',a**2)

a_sum = np.sum(a,1)
print('a_sum',a_sum)

a_sum_reshape = a_sum.reshape(-1,1)

print('a_sum_reshape',a_sum_reshape)

print('a+a',a_sum_reshape + a_sum)

aa = np.dot(a,a.T)

print('aa',aa)

aa1 = a@a.T
print('aa1',aa1)

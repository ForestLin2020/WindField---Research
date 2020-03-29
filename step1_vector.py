import matplotlib.pyplot as plt
import numpy as np

# # Original point
# X = np.array((0))
# Y= np.array((0))
# print('X',X)
#
# # End direction
# U = np.array((1))
# V = np.array((1))
#
# # create a person and give it a job
# fig, ax = plt.subplots()
# q = ax.quiver(X, Y, U, V, units='xy', scale=1)
# # q = ax.quiver(X, Y, U, V)
#
#
# # draw the grid
# plt.grid()
#
# # set the aspect
# ax.set_aspect('equal')
#
# # range
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
#
# plt.title('a singal vector',fontsize=10)
#
# # plt.savefig('how_to_plot_a_vector_in_matplotlib_fig3.png', bbox_inches='tight')
# plt.show()
# plt.close()


######################[2]############################
fig, ax = plt.subplots()

x_pos = [0, 1, 2]
y_pos = [0, 1, 2]
x_direct = [1, -2]
y_direct = [1, -2]
# print('x_pos',x_pos)
# print('x_direct',x_direct)
# print('y_direct',y_direct)

q = ax.quiver(x_pos,y_pos, units='xy', scale=1)

print('q= ', q)
ax.axis([-5, 5, -5, 5])
plt.grid()
ax.set_aspect('equal')


plt.show()


####################[3]###########################


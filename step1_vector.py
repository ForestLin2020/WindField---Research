import matplotlib.pyplot as plt
import numpy as np

# Original point
X = np.array((0))
Y= np.array((0))

# End direction
U = np.array((2))
V = np.array((-2))

# create a person and give it a job
fig, ax = plt.subplots()
# q = ax.quiver(X, Y, U, V,units='xy' ,scale=3)
q = ax.quiver(X, Y, U, V)


# draw the grid
plt.grid()

# set the aspect
ax.set_aspect('equal')

# range
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.title('a singal vector?',fontsize=10)

# plt.savefig('how_to_plot_a_vector_in_matplotlib_fig3.png', bbox_inches='tight')
plt.show()
plt.close()
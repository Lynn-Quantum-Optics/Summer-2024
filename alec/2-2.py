import numpy as np
import matplotlib.pyplot as plt
from random_states import random_prob_vector, SX, SY, SZ

# define the random unit vector function that is discussed in 2.1

def random_unit_vector_prob(dim):
    x = np.sqrt(random_prob_vector(dim)) * np.exp(2j*np.pi*np.random.rand(dim))
    return x

# generate a bunch of states using that function

N = 10000
states = np.array([random_unit_vector_prob(2) for _ in range(N)])

# calculate positions on the bloch sphere

pos = np.array([
    [s.conj().T @ SX @ s,
     s.conj().T @ SY @ s,
     s.conj().T @ SZ @ s] for s in states]).real.T

# +++ FIGURE 1 +++

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pos[0], pos[1], pos[2], s=0.8, alpha=0.5)
ax.set_xlabel(r'$\langle \sigma_x \rangle$')
ax.set_ylabel(r'$\langle \sigma_y \rangle$')
ax.set_zlabel(r'$\langle \sigma_z \rangle$')
fig.canvas.manager.set_window_title('method2_pure_bloch_sphere')
plt.show()

# +++ FIGURE 2 +++

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(311)
ax.hist(pos[0], bins=30)
ax.set_title(r'$\langle \sigma_x\rangle$')
ax = fig.add_subplot(312)
ax.hist(pos[1], bins=30)
ax.set_title(r'$\langle \sigma_y\rangle$')
ax = fig.add_subplot(313)
ax.hist(pos[2], bins=30)
ax.set_title(r'$\langle \sigma_z\rangle$')
fig.tight_layout()

fig.canvas.manager.set_window_title('method2_pure_histograms')
plt.show()

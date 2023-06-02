import numpy as np
import matplotlib.pyplot as plt
import random_states
from random_states import SX,SY,SZ

# 
def random_probs_vector(n=3):
    # generate a random vector with n elements that sum to 1
    vec = []
    while len(vec) < n-1:
        vec.append(np.random.rand()*(1-np.sum(vec)))
    # add last element so vector sum is unity
    vec.append(1-np.sum(vec))
    # randomly permute the vector
    np.random.shuffle(vec)
    return np.array(vec)

method0 = np.array([random_probs_vector() for i in range(10000)]).T

method1 = np.random.rand(3,10000)
method1 = method1/np.sum(method1, axis=0)

method2 = (np.random.randn(3,10000))
method2 = method2/np.linalg.norm(method2, axis=0)
method2 = method2**2

# s = 0.5
# alpha = 0.1
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(211, projection='3d')
# ax.scatter(method0[0], method0[1], method0[2], s=s, alpha=alpha, label='Even')
# # plt.title('Method 0')
# # plt.show()
# # ax = fig.add_subplot(projection='3d')
# # ax.scatter(method1[0], method1[1], method1[2], s=s, alpha=alpha, color=(0,1,0), label='Uniform')
# # plt.title('Method 1')
# # plt.show()
# ax = fig.add_subplot(212, projection='3d')
# ax.scatter(method2[0], method2[1], method2[2], color=(1,0,0), s=s, alpha=alpha, label='Gaussian')
# # plt.title('Method 2')
# plt.legend()
# plt.show()


# fig = plt.figure(figsize=(8,8))
# # plt.scatter(method0[0], method0[1], s=0.9, alpha=0.5)
# ax = fig.add_subplot(311)
# ax.set_title('Even')
# ax.hist(np.concatenate(method0), bins=30)

# ax = fig.add_subplot(312)
# ax.set_title('Uniform')
# ax.hist(np.concatenate(method1), bins=30)

# ax = fig.add_subplot(313)
# ax.hist(np.concatenate(method2), bins=30)
# ax.set_title('Gaussian')

# plt.show()

states = np.sqrt(np.array([random_states.random_prob_vector(2) for _ in range(10000)])) * np.exp(1j*np.random.rand(10000,2)*2*np.pi)

xs = np.array([s.conj().T @ SX @ s for s in states]).real
ys = np.array([s.conj().T @ SY @ s for s in states]).real
zs = np.array([s.conj().T @ SZ @ s for s in states]).real

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, s=0.8, alpha=0.5)
plt.show()
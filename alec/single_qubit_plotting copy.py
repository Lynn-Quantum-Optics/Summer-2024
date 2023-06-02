import numpy as np
import matplotlib.pyplot as plt
import random_states


N = 10000

SX = np.array([[0,1],[1,0]])
SY = np.array([[0,-1j],[1j,0]])
SZ = np.array([[1,0],[0,-1]])
# rhos = np.array([random_states.RDM1(2) for _ in range(N)])
rhos = np.array([random_states.RDM4(4,4) for _ in range(N)])
# rhos = np.array([random_states.RDM3_safe() for _ in range(N)])
basis = random_states.TWO_QUBIT_PAULI_BASIS.T

def get_coords(r):
    return np.array([np.trace(b @ r) for b in basis]).real

coords = np.array([get_coords(rho) for rho in rhos])
radii = np.linalg.norm(coords, axis=1)

# xs = np.array([np.trace(SX @ rho) for rho in rhos]).real
# ys = np.array([np.trace(SY @ rho) for rho in rhos]).real
# zs = np.array([np.trace(SZ @ rho) for rho in rhos]).real

''' Purity by Radius plot '''
purity = np.array([np.trace(rho @ rho) for rho in rhos])

plt.scatter(radii**2, purity, alpha=0.5)
plt.xlabel('Radius')
plt.ylabel('Purity')
plt.show()


''' Frequency / Radius by Radius plot
bins = np.linspace(0.01,1,100)
counts = np.zeros_like(bins)
for r in radii:
    counts[np.argmin(np.abs(r - bins))] += 1
counts /= bins**2
plt.bar(bins, counts, width=0.01)
plt.xlabel('Radius')
plt.ylabel('Frequency / Radius')
plt.show()
'''

'''
xs_tp = xs[(xs > 0) * (ys > 0) * (zs > 0)]
ys_tp = ys[(xs > 0) * (ys > 0) * (zs > 0)]
zs_tp = zs[(xs > 0) * (ys > 0) * (zs > 0)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs_tp,ys_tp,zs_tp,alpha=0.5)
ax.set_xlabel('<σx>')
ax.set_ylabel('<yx>')
ax.set_zlabel('<zx>')
plt.show()
'''
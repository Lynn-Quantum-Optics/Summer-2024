import numpy as np
import matplotlib.pyplot as plt
import random_states
import concurrence
import scipy.linalg as la

N = 5000

rhos1 = np.array([random_states.RDM4(4,2) for _ in range(N)])
conc1 = np.array([concurrence.concurrence(rho) for rho in rhos1])
purity1 = np.array([np.trace(rho @ rho) for rho in rhos1])
ent_purity1 = np.array([np.trace(rho @ rho) for (rho,c) in zip(rhos1, conc1) if c > 0])

rhos2 = np.array([random_states.RDM4(4,4) for _ in range(N)])
conc2 = np.array([concurrence.concurrence(rho) for rho in rhos2])
purity2 = np.array([np.trace(rho @ rho) for rho in rhos2])
ent_purity2 = np.array([np.trace(rho @ rho) for (rho,c) in zip(rhos2, conc2) if c > 0])

rhos3 = np.array([random_states.RDM4(4,8) for _ in range(N)])
conc3 = np.array([concurrence.concurrence(rho) for rho in rhos3])
purity3 = np.array([np.trace(rho @ rho) for rho in rhos3])
ent_purity3 = np.array([np.trace(rho @ rho) for (rho,c) in zip(rhos3, conc3) if c > 0])

pct_ent_1 = (1 - np.sum(conc1 == 0)/N)*100
pct_ent_2 = (1 - np.sum(conc2 == 0)/N)*100
pct_ent_3 = (1 - np.sum(conc3 == 0)/N)*100

fig = plt.figure()

# concurrence

ax = fig.add_subplot(331)
ax.hist(conc1, bins=np.linspace(0.01,1,100))
ax.set_xlabel(f'Concurrence ({pct_ent_1:.2f}% shown)')
ax.set_title(f'Method 1')

ax = fig.add_subplot(332)
ax.hist(conc2, bins=np.linspace(0.01,1,100))
ax.set_xlabel(f'Concurrence ({pct_ent_2:.2f}% shown)')
ax.set_title(f'Method 2')

ax = fig.add_subplot(333)
ax.hist(conc3, bins=np.linspace(0.01,1,100))
ax.set_xlabel(f'Concurrence ({pct_ent_3:.2f}% shown)')
ax.set_title(f'Method 3')

# purity

ax = fig.add_subplot(334)
ax.hist(purity1, bins=np.linspace(0,1,100))
ax.set_xlabel('Purity')

ax = fig.add_subplot(335)
ax.hist(purity2, bins=np.linspace(0,1,100))
ax.set_xlabel('Purity')

ax = fig.add_subplot(336)
ax.hist(purity3, bins=np.linspace(0,1,100))
ax.set_xlabel('Purity')

# entangled purity

ax = fig.add_subplot(337)
ax.hist(ent_purity1, bins=np.linspace(0,1,100))
ax.set_xlabel('Purity of Entangled States')

ax = fig.add_subplot(338)
ax.hist(ent_purity2, bins=np.linspace(0,1,100))
ax.set_xlabel('Purity of Entangled States')

ax = fig.add_subplot(339)
ax.hist(ent_purity3, bins=np.linspace(0,1,100))
ax.set_xlabel('Purity of Entangled States')

fig.tight_layout(pad=1)
plt.show()

# def fidelity(r1, r2):
#     return np.trace(la.sqrtm(la.sqrtm(r1) @ r2 @ la.sqrtm(r1)))**2

# dist_1 = []
# dist_2 = []
# dist_3 = []
# for i in range(N):
#     for j in range(i):
#         dist_1.append(1-fidelity(rhos1[i], rhos1[j]))
#         dist_2.append(1-fidelity(rhos2[i], rhos2[j]))
#         dist_3.append(1-fidelity(rhos3[i], rhos3[j]))

# BINS = 70

# fig = plt.figure()
# ax = fig.add_subplot(221)
# ax.hist(dist_1, bins=BINS)
# ax.set_xlim(0,1)
# ax.set_title('METHOD 1')
# ax.set_xlabel('pairwise distance')
# ax.set_ylabel('Count')

# ax = fig.add_subplot(222)
# ax.hist(dist_2, bins=BINS)
# ax.set_xlim(0,1)
# ax.set_title('METHOD 2')
# ax.set_xlabel('pairwise distance')
# ax.set_ylabel('Count')

# ax = fig.add_subplot(223)
# ax.hist(dist_3, bins=BINS)
# ax.set_xlim(0,1)
# ax.set_title('METHOD 3')
# ax.set_xlabel('pairwise distance')
# ax.set_ylabel('Count')
# plt.show()
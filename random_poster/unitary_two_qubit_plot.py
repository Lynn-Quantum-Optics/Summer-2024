from qo_tools import *
from unitary import *
from simplex import *
from random_states import *
from plotting import *
from tqdm import tqdm
import numpy as np

# make states
N = 100000
states_2qb = [random_mixed_unitary(4) for _ in tqdm(range(N))]

# take expectation value measurements
meas = {}
for label, basis in [('zi', np.kron(SZ,ID)), ('iz', np.kron(ID, SZ)), ('zz', np.kron(SZ,SZ)), ('xx', np.kron(SX, SX))]:
    meas[label] = [expectation_value(s, basis) for s in states_2qb]

# figure setup
fig = plt.figure(figsize=(6, 4))
fig.suptitle('Expectation Value Probability Densities')

# bins for histograms
BINS = np.linspace(-1,1,101)

# first plot for zi/iz
ax = fig.add_subplot(1,2,1)
ax.hist(meas['zi'], bins=BINS, density=True, label=r'$\langle\sigma_{zi}\rangle$', alpha=0.5)
ax.hist(meas['iz'], bins=BINS, density=True, label=r'$\langle\sigma_{iz}\rangle$', alpha=0.5)
ax.legend()
ax.set_xlabel('Expectation Value')
ax.set_ylabel('Probability Density')
ax.set_title('ZI and IZ')

# second plot for zz/xx
ax = fig.add_subplot(1,2,2)
ax.hist(meas['zz'], bins=BINS, density=True, label=r'$\langle\sigma_{zz}\rangle$', alpha=0.5)
ax.hist(meas['xx'], bins=BINS, density=True, label=r'$\langle\sigma_{xx}\rangle$', alpha=0.5)
ax.legend()
ax.set_xlabel('Expectation Value')
ax.set_ylabel('Probability Density')
ax.set_title('ZZ and XX')

fig.tight_layout()
plt.savefig('unitary_two_qubit_demo_plots.png', dpi=600)


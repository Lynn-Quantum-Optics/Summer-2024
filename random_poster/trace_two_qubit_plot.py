from qo_tools import *
from unitary import *
from simplex import *
from random_states import *
from plotting import *
from tqdm import tqdm
import numpy as np

N = 100000
states_2qb = [random_mixed_trace(log2n=2, log2start=4) for _ in tqdm(range(N))]
meas = {}
bases = {}

# measure in all the different bases
for al, a in zip('ixyz', [ID, SX, SY, SZ]):
    for bl, b in zip('ixyz', [ID, SX, SY, SZ]):
        meas[al+bl] = []
        bases[al+bl] = np.kron(a,b)

for s in tqdm(states_2qb):
    for l in meas:
        meas[l].append(expectation_value(s,bases[l]))

# setup the figure
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)

# plot histograms
BINS = np.linspace(-1,1,101)
for basis in 'ix,iy,iz,xi,xx,xy,xz,yi,yx,yy,yz,zi,zx,zy,zz'.split(','):
    ax.hist(meas[basis], bins=BINS, density=True, alpha=0.05, label=basis)

# plot the theoretical density curve
def plot_hypersphere_density_curve(res=101, ax=ax, color='k', label='Theoretical'):
    x = np.linspace(-1,1,res)
    r = np.sqrt(1-x**2)
    v14 = np.pi**7/5040*np.power(r,14)
    dx = x[1]-x[0]
    dv = v14*dx
    density = dv/(dx*np.sum(dv))
    ax.plot(x, density, color=color, label=label)

plot_hypersphere_density_curve()

ax.legend()
ax.set_title('Expectation Value Probability Densities\nwith Theoretical 15-Dimensional Hypersphere Density')
ax.set_xlabel('Expectation Value')
ax.set_ylabel('Density')
fig.tight_layout()

for l in meas:
    print(f'{l}: {np.std(meas[l]):.3f}')

plt.savefig('trace_two_qubit_prob.png', dpi=600)

plt.show()

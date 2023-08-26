from qo_tools import *
from unitary import *
from simplex import *
from random_states import *
from plotting import *
from tqdm import tqdm
import numpy as np

N = 100000
two_qubit = True
states_2qb = [random_mixed_unitary(4) for _ in tqdm(range(N))]
# states_1qb = [random_mixed_unitary(2) for _ in tqdm(range(N))]
# states_2qb = [random_mixed_trace(2) for _ in tqdm(range(N))]
# states_1qb = [random_mixed_trace(1) for _ in tqdm(range(N))]

meas = {}

# for al, a in zip('xyz', [SX, SY, SZ]):
#     meas[al] = [expectation_value(s,a) for s in states_1qb]

for al, a in zip('ixyz', [ID, SX, SY, SZ]):
    for bl, b in zip('ixyz', [ID, SX, SY, SZ]):
        meas[al+bl] = [expectation_value(s,np.kron(a,b)) for s in states_2qb]

# groupings = [('xi,xx,xy,xz,yi,yx,yy,yz,x,y,z', 0.125), ('ix,iy,zx,zy,x,y,z', 0.25), ('zz,iz,x,y,z', 0.25), ('zi,x,y,z', 0.25)]


groupings = [('xi,xx,xy,xz,yi,yx,yy,yz', 0.125, 3), ('ix,iy,zx,zy', 0.25, 2.5), ('zz,iz', 0.5, 2.25), ('zi', 1, 2.0)]


fig = plt.figure(figsize=(8, 8))

for i, (group, alph, ymax) in enumerate(groupings):
    ax = fig.add_subplot(2,2,i+1)
    for l in group.split(','):
        ax.hist(meas[l], bins=np.linspace(-1,1,101), density=True, alpha=alph, label=l)
    ax.legend()
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Expectation Value')
    ax.set_ylabel('Density')


for l in meas:
    print(f'{l}: {np.std(meas[l]):.3f}')

fig.suptitle('Expectation Value Probability Densities')
fig.tight_layout()
# plt.savefig('maziero_exp_prob.png', dpi=600)
plt.show()

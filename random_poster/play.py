from qo_tools import *
from unitary import *
from simplex import *
from random_states import *
from plotting import *
from tqdm import tqdm

N = 100000

states = [random_mixed_trace(2) for _ in tqdm(range(N))]
# states = [random_mixed_unitary(4, unitary_method='roik') for _ in tqdm(range(N))]
pauli_histograms_2qubit(states, bins=100, figsize=(16,8))
plt.show()
# dist = 'uniform re-im'

# # generate states
# states = []
# print('Generating states...')
# for _ in tqdm(range(N+1)):
#     states.append(random_mixed_unitary(8, unitary_method='maziero'))
# # calculate distances
# print('Calculating distances...')
# dists = []
# for i in tqdm(range(N)):
#     dists.append(HS_dist(states[i], states[i+1]))

# plt.hist(dists, bins=70, density=True, stacked=True)
# plt.show()

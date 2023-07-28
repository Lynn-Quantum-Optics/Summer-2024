from random_states import random_mixed_unitary, random_mixed_trace, random_mixed_op
from plotting import pauli_histograms_1qubit, plot_on_bloch_sphere
from matplotlib import pyplot as plt
from tqdm import tqdm

# generate the states
NUM = 100000

file_names = [
    'unitary',
    'trace',
    'op']

methods = [
    'Unitary',
    'Trace',
    'Over Parameterized']

gen_funcs = [
    lambda : random_mixed_unitary(2, simplex_method='stick', unitary_method='maziero'),
    lambda : random_mixed_trace(log2n=1, log2start=2),
    lambda : random_mixed_op(2, dist='uniform re-im')]

for name, f, fname in zip(methods, gen_funcs, file_names):
    states = [f() for _ in tqdm(range(NUM))]
    plot_on_bloch_sphere(states, title=f'{name}', figsize=(4,4), s=0.5, alpha=0.05)
    plt.savefig(f'mixed_one_qubit_plots/{fname}.png')

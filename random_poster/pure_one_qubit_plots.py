from random_states import random_pure
from plotting import pauli_histograms_1qubit, plot_on_bloch_sphere
from matplotlib import pyplot as plt

# generate the states
NUM = 10000
gauss_states = [random_pure(2, simplex_method='gauss') for _ in range(NUM)]
stick_states = [random_pure(2) for _ in range(NUM)]

# GAUSS BLOCH
plot_on_bloch_sphere(gauss_states, title=f'Bloch Sphere Plot\n{NUM} random states; Gaussian simplex', figsize=(4,4), s=0.5, alpha=0.3)
plt.savefig('./pure_one_qubit_plots/bloch_sphere_gauss.png', dpi=400)

# GAUSS HIST
pauli_histograms_1qubit(gauss_states, title=f'Distribution of Bloch Coordinates\n{NUM} random states; Gaussian simplex', figsize=(4,8))
plt.savefig('./pure_one_qubit_plots/histograms_gauss.png', dpi=400)

# STICK BLOCH
plot_on_bloch_sphere(stick_states, title=f'Bloch Sphere Plot\n{NUM} random states; stick-breaking simplex', figsize=(4,4), s=0.5, alpha=0.3)
plt.savefig('./pure_one_qubit_plots/bloch_sphere_stick.png', dpi=400)

# STICK HIST
pauli_histograms_1qubit(stick_states, title=f'Distribution of Bloch Coordinates\n{NUM} random states; stick-breaking simplex', figsize=(4,8))
plt.savefig('./pure_one_qubit_plots/histograms_stick.png', dpi=400)
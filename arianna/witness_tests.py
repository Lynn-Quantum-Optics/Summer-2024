from qutip import *
from math import sqrt, cos, sin
import numpy as np
from gen_state import make_W_list
from rho_methods_ONLYGOOD import compute_witnesses, get_rho
import matplotlib.pyplot as plt

"""
Uses compute_witnesses function from Oscar's rho_methods to find the expected witness values for a variety of states
The witness expectation values are displayed graphically for all 15 witnesses.
"""



# basis states to play with
# Basis Notes: 
H = basis(2,0) # ket 1
V = basis(2,1) # ket 2
D = 1/sqrt(2) * (H + V)
A = 1/sqrt(2) * (H - V)

# Bell state Psi = 1/sqrt(2) (|HV> +/- |VH>)
psi_p = 1/sqrt(2)*(tensor(H,V) + tensor(V,H))
psi_n = 1/sqrt(2)*(tensor(H,V) - tensor(V,H))

# Bell state Phi = 1/sqrt(2) (|HH> +/- |VV>)
phi_p = 1/sqrt(2)*(tensor(H,H) + tensor(V,V))
phi_n = 1/sqrt(2)*(tensor(H,H) - tensor(V,V))

eta = np.radians(45)
chi = np.radians(90)


# test which bell states are witnessed by which operators
test_1 = psi_p # entangled
test_2 = psi_n # entangled
test_3 = phi_p # entangled
test_4 = phi_n # entangled
test_5 = phi_p + phi_n # not entangled
test_6 = psi_p + psi_n # not entangled
test_7 = 1/sqrt(2) * (phi_p + psi_p) # not entangled
test_8 = phi_n + psi_n # not entangled
test_9 = 1/sqrt(2) * (psi_p + 1j*psi_n)#(5*phi_p + 1j*psi_n) # entangled
test_10 = np.cos(eta)*phi_p + np.exp(1j*chi)*np.sin(eta)*phi_n # entangled

test_states = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10]
test_rho = [ket2dm(state).full() for state in test_states]


state_wit_vals = {} # stores all the states corresponding witness measurements. 
for i in range(len(test_rho)):
    state_wit_vals[i] = [[] for i in range(15)]


num_tries = 1
for j in range(num_tries):
    for i,rho in enumerate(test_rho):
        wit_list = compute_witnesses(rho)
        for m in range(len(wit_list)):
            state_wit_vals[i][m] += [wit_list[m]]


# Create a figure with 2 rows and 5 columns of subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Test State Witness Values')  # Add a title to the whole figure

# the x values are the different witnesses possible to run on the state
x = range(16)[1:]

# Create 10 subplots
for i, ax in enumerate(axes.flatten()):
    y = []
    y_err = []
    for j in range(15):
        y += [np.mean(state_wit_vals[i][j])]
        y_err += [np.std(state_wit_vals[i][j])/np.sqrt(num_tries)]

    ax.scatter(x, y, label=f'Test {i+1}')  # Plot the data
    ax.errorbar(x,y,y_err, label=f'Error {i+1}', fmt='o')
    ax.set_title(f'Test State {i+1}')  # Set subplot titles
    ax.set_xlabel('Witness')  # Set X-axis label
    ax.set_ylabel('Witness Value')  # Set Y-axis label

plt.tight_layout()  # Adjust subplot spacing
plt.show()


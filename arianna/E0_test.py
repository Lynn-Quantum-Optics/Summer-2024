from qutip import *
from math import sqrt, cos, sin
import numpy as np
from gen_state import make_W, make_E0, make_WPrimes
import matplotlib.pyplot as plt

# basis states to play with
# Basis Notes: 
H = basis(2,0) # ket 1
V = basis(2,1) # ket 2

# Bell state Psi = 1/sqrt(2) (|HV> +/- |VH>)
psi_p = 1/sqrt(2)*(tensor(H,V) + tensor(V,H))
psi_n = 1/sqrt(2)*(tensor(H,V) - tensor(V,H))

# Bell state Phi = 1/sqrt(2) (|HH> +/- |VV>)
phi_p = 1/sqrt(2)*(tensor(H,H) + tensor(V,V))
phi_n = 1/sqrt(2)*(tensor(H,H) - tensor(V,V))



# test which bell states are witnessed by which operators
test_1 = psi_p
test_2 = psi_n
test_3 = phi_p
test_4 = phi_n
test_5 = phi_p + phi_n
test_6 = psi_p + psi_n
test_7 = phi_p + psi_p
test_8 = phi_n + psi_n
test_9 = phi_p + 1j*psi_n
test_10 = phi_n + 1j*psi_p

test_states = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10]
test_rho = [ket2dm(state) for state in test_states]

a_range = np.linspace(0.01,0.99,100)
min_witness_val = 0

state_wit_vals = {} # stores all the states corresponding witness measurements. 
                    # Each state gets a list of length 6 for all the witnesses, each with a (a,b, val) tuple


w_list = []


# add the scipy minimized version of witnesses here at some point
for i in range(len(test_rho)):
    state_wit_vals[i] = 6*[(0,0,0)]
    state = test_rho[i]

    for j in range(6):
        curr_w_val = 1e10
        curr_a = 0
        curr_b = 0

        for a in a_range:
            b =sqrt(1-a**2)

            w_list = make_W(a,b)
            w_val = (w_list[j]*state).tr()

            if w_val < curr_w_val:
                curr_w_val = w_val
                curr_a = a
                curr_b = b

        if curr_w_val < 0:
            state_wit_vals[i][j] = (curr_a, curr_b, curr_w_val)
        
print(state_wit_vals)


# Create a figure with 2 rows and 5 columns of subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Test State Witness Values')  # Add a title to the whole figure

# Generate some sample data
x = range(6)

# Create 10 subplots
for i, ax in enumerate(axes.flatten()):
    y = []
    for j in range(6):
        y += [state_wit_vals[i][j][2]]

    ax.scatter(x, y, label=f'Test {i+1}')  # Plot the data
    ax.set_title(f'Test State {i+1}')  # Set subplot titles
    ax.set_xlabel('Witness')  # Set X-axis label
    ax.set_ylabel('Witness Value')  # Set Y-axis label

plt.tight_layout()  # Adjust subplot spacing
plt.show()

    
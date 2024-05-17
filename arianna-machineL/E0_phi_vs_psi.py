from qutip import *
from math import sqrt, cos, sin
import numpy as np
from gen_state import make_E0, make_W_list
from rho_methods_ONLYGOOD import compute_witnesses, get_rho
import matplotlib.pyplot as plt

chi_range = np.linspace(0,np.pi/2, 6) # for a hundred values of chi
eta_L = [np.pi / 4, np.pi/6] # eta should always be 30 or 45 degrees

curr_min_exp_phi_45 = [1e10 for i in range(15)] # initialize the tracker for the current minimum expectation value
curr_min_exp_psi_45 = [1e10 for i in range(15)]
curr_min_exp_phi_30 = [1e10 for i in range(15)] # initialize the tracker for the current minimum expectation value
curr_min_exp_psi_30 = [1e10 for i in range(15)]
record_mins_phi = {30:[0 for i in range(15)],45:[0 for i in range(15)]}
record_mins_psi = {30:[0 for i in range(15)],45:[0 for i in range(15)]}

for eta in eta_L:
    for chi in chi_range:
        # make the phi based E0 state and the psi based
        E0_phi = make_E0(eta, chi, "phi")
        E0_psi = make_E0(eta, chi, "psi")

        # convert to oscar's code 
        E0_phi = E0_phi.full() # E0 prime
        E0_psi = E0_psi.full() # E0 

        # compute the min W for this state
        phi_wit_exp = compute_witnesses(E0_phi, return_all=True)
        psi_wit_exp = compute_witnesses(E0_psi,return_all=True)

        # update min 
        for i in range(len(curr_min_exp_phi_45)):
            if phi_wit_exp[i] < curr_min_exp_phi_45[i] and eta == np.pi / 4:
                curr_min_exp_phi_45[i] = phi_wit_exp[i]
                record_mins_phi[45][i] = [chi] 
            if psi_wit_exp[i] < curr_min_exp_psi_45[i] and eta == np.pi / 4:
                curr_min_exp_psi_45[i] = psi_wit_exp[i]
                record_mins_psi[45][i] = [chi] 
            if phi_wit_exp[i] < curr_min_exp_phi_30[i] and eta == np.pi / 6:
                curr_min_exp_phi_30[i] = phi_wit_exp[i]
                record_mins_phi[30][i] = [chi] 
            if psi_wit_exp[i] < curr_min_exp_psi_30[i] and eta == np.pi / 6:
                curr_min_exp_psi_30[i] = psi_wit_exp[i]
                record_mins_psi[30][i] = [chi] 

print("phi", record_mins_phi)
print("psi",record_mins_psi)
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle('E0 for Phi and Psi Expectation Value') 

x_W_45 = range(7)[1:]
y1_W_45 = curr_min_exp_phi_45[:6]
y2_W_45 = curr_min_exp_psi_45[:6]

x_Wp_45 = range(16)[7:]
y1_Wp_45 = curr_min_exp_phi_45[6:]
y2_Wp_45 = curr_min_exp_psi_45[6:]

ax[0][0].scatter(x_W_45, y1_W_45, color='red', label='E0 Phi Expectation, 45 W', marker='o')
ax[0][0].scatter(x_W_45, y2_W_45, color='blue', label='E0 Psi Expectation, 45 W', marker='x')
ax[0][0].set_title('W Expectation Values, 45')

ax[0][1].scatter(x_Wp_45, y1_Wp_45, color='red', label='E0 Phi Expectation, 45 Wp', marker='o')
ax[0][1].scatter(x_Wp_45, y2_Wp_45, color='blue', label='E0 Psi Expectation, 45 Wp', marker='x')
ax[0][1].set_title("W' Expectation Values, 45")

x_W_30 = range(7)[1:]
y1_W_30 = curr_min_exp_phi_30[:6]
y2_W_30 = curr_min_exp_psi_30[:6]

x_Wp_30 = range(16)[7:]
y1_Wp_30 = curr_min_exp_phi_30[6:]
y2_Wp_30 = curr_min_exp_psi_30[6:]

ax[1][0].scatter(x_W_30, y1_W_30, color='red', label='E0 Phi Expectation, 30 W', marker='o')
ax[1][0].scatter(x_W_30, y2_W_30, color='blue', label='E0 Psi Expectation, 30 W', marker='x')
ax[1][0].set_title('W Expectation Values, 30')

ax[1][1].scatter(x_Wp_30, y1_Wp_30, color='red', label='E0 Phi Expectation, 30 Wp', marker='o')
ax[1][1].scatter(x_Wp_30, y2_Wp_30, color='blue', label='E0 Psi Expectation, 30 Wp', marker='x')
ax[1][1].set_title("W' Expectation Values, 30")


plt.legend()
plt.show()

"""
Min values for each witness
phi {30: [[0.0], [0.09617120368132019], [0.0641141357875468], [0.0641141357875468], [0.0641141357875468], [0.0641141357875468], [0.0], [1.5707963267948966], [1.5707963267948966], [0.09617120368132019], [1.1540544441758422], [1.218168579963389], [0.6731984257692414], [0.09617120368132019], [0.9296549689194286]], 45: [[0.0], [0.0], [0.6411413578754679], [0.6411413578754679], [0.6411413578754679], [0.6411413578754679], [0.0], [1.5707963267948966], [1.5707963267948966], [1.1540544441758422], [1.1540544441758422], [0.0], [1.1540544441758422], [1.1540544441758422], [0.0]]}
psi {30: [[0.09617120368132019], [0.0], [0.09617120368132019], [0.6731984257692414], [0.09617120368132019], [0.6731984257692414], [1.5707963267948966], [0.0], [1.5707963267948966], [0.09617120368132019], [0.6731984257692414], [0.32057067893773394], [0.09617120368132019], [0.6731984257692414], [0.9296549689194286]], 45: [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.5707963267948966], [0.0], [1.5707963267948966], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]}
"""

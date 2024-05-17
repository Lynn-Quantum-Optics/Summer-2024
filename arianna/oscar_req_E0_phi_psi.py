from qutip import *
from math import sqrt, cos, sin
import numpy as np
from gen_state import make_E0, make_W_list
from rho_methods_ONLYGOOD import compute_witnesses, get_rho
import matplotlib.pyplot as plt

chi_range = np.linspace(0,np.pi/2, 6) # for a hundred values of chi
eta_L = [np.pi / 4, np.pi/6] # eta should always be 30 or 45 degrees

phi_wit_exp_W = [0]*len(chi_range)
psi_wit_exp_W = [0]*len(chi_range)
phi_wit_exp_Wp = [0]*len(chi_range)
psi_wit_exp_Wp = [0]*len(chi_range)

eta = np.deg2rad(30)
chi = np.deg2rad(15)

E0_phi = make_E0(eta, chi, "phi")
E0_psi = make_E0(eta, chi, "psi")

E0_phi = E0_phi.full() # E0 prime
E0_psi = E0_psi.full()

print(compute_witnesses(E0_phi))
print(compute_witnesses(E0_psi))

# for eta in eta_L:
#     for i in range(len(chi_range)):
#         # make the phi based E0 state and the psi based
#         E0_phi = make_E0(eta, chi_range[i], "phi")
#         E0_psi = make_E0(eta, chi_range[i], "psi")

#         # convert to oscar's code 
#         E0_phi = E0_phi.full() # E0 prime
#         E0_psi = E0_psi.full() # E0 

#         # compute the min W for this state



#         phi_wit_exp_W[i] = min(compute_witnesses(E0_phi)[:6])
#         psi_wit_exp_W[i] = min(compute_witnesses(E0_psi)[:6])
#         phi_wit_exp_Wp[i] = min(compute_witnesses(E0_phi)[6:])
#         psi_wit_exp_Wp[i] = min(compute_witnesses(E0_psi)[6:])

#     # record for diff eta 
#     if eta == np.pi/4:
#         w_exp_45_phi = phi_wit_exp_W
#         w_exp_45_psi = psi_wit_exp_W
#         wp_exp_45_phi = phi_wit_exp_Wp
#         wp_exp_45_psi = psi_wit_exp_Wp
#     else:
#         w_exp_30_phi = phi_wit_exp_W
#         w_exp_30_psi = psi_wit_exp_W
#         wp_exp_30_phi = phi_wit_exp_Wp
#         wp_exp_30_psi = psi_wit_exp_Wp


# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# fig.suptitle('E0 for Phi and Psi Expectation Value') 


# ax[0].scatter(chi_range, w_exp_45_psi, color='red', label='E0 Phi Expectation, 45 W', marker='o')
# ax[0].scatter(chi_range, wp_exp_45_psi, color='blue', label='E0 Phi Expectation, 45 Wp', marker='x')
# ax[0].set_title('W Expectation Values, 45')

# ax[1].scatter(chi_range, w_exp_30_psi, color='red', label='E0 Phi Expectation, 30 W', marker='o')
# ax[1].scatter(chi_range, wp_exp_30_psi, color='blue', label='E0 Phi Expectation, 30 Wp', marker='x')
# ax[1].set_title("W Expectation Values, 30")



# plt.legend()
# plt.show()

"""
Min values for each witness
phi {30: [[0.0], [0.09617120368132019], [0.0641141357875468], [0.0641141357875468], [0.0641141357875468], [0.0641141357875468], [0.0], [1.5707963267948966], [1.5707963267948966], [0.09617120368132019], [1.1540544441758422], [1.218168579963389], [0.6731984257692414], [0.09617120368132019], [0.9296549689194286]], 45: [[0.0], [0.0], [0.6411413578754679], [0.6411413578754679], [0.6411413578754679], [0.6411413578754679], [0.0], [1.5707963267948966], [1.5707963267948966], [1.1540544441758422], [1.1540544441758422], [0.0], [1.1540544441758422], [1.1540544441758422], [0.0]]}
psi {30: [[0.09617120368132019], [0.0], [0.09617120368132019], [0.6731984257692414], [0.09617120368132019], [0.6731984257692414], [1.5707963267948966], [0.0], [1.5707963267948966], [0.09617120368132019], [0.6731984257692414], [0.32057067893773394], [0.09617120368132019], [0.6731984257692414], [0.9296549689194286]], 45: [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.5707963267948966], [0.0], [1.5707963267948966], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]}
"""

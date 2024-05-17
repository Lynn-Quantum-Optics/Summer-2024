from gen_state import make_W, make_E0, make_WPrimes
import numpy as np
from qutip import *

#
# Makes witnesses
#

# Basis Notes: 
H = basis(2,0) # ket 1
V = basis(2,1) # ket 2

eta = np.pi/4
chi = np.pi/4

# make a state to be witnessed, a state hopefully not witnessed
rho_E0 = make_E0(eta,chi) # not witnessable density matrix
rho_E0_psi2 = make_E0(eta, 0) # ?witnessable density matrix

test_state = 1/np.sqrt(2)*(tensor(H,H) + tensor(V,V))
rho_test = ket2dm(test_state)

# make list of witnesses with a,b
# witnessL = make_W(0.5,0.2) # not sure how to pick a and b
# E0_ent = {True:[], False:[]}
# test = {True:[], False:[]}
# E0_psi_2 = {True:[], False:[]}

a_range = np.linspace(0.01,0.99,10)

# for a in a_range:
#     bL = [np.sqrt(1-a**2), -np.sqrt(1- a**2)]
#     for b in bL:
#         witnessL = make_W(a,b)
#         for i in range(len(witnessL)):
#             W = witnessL[i]
            
#             # get witness values
#             val_E0 = (W*(rho_E0)).tr()
#             val_test = (W*(rho_test)).tr()
#             val_E0_psi2 = (W*(rho_E0_psi2)).tr()
            
#             # if the values are less than 0 add them
#             E0_ent[bool(val_E0 < 0)] += [(a,b, i)]
#             test[bool(val_test < 0)] += [(a,b, i)]
#             E0_psi_2[bool(val_E0_psi2 < 0)] += [(a,b, i)]

# print(f"Less than 0 E0 state: {E0_ent[True]}")
# print(f"Less than 0 test entangled state: {len(test[True])}")
# print(f"Less than 0 E0 state made to resemble psi2: {len(E0_psi_2[True])}")


#
# Next test whether E0 is witnessed for a range of eta, chi
# compare W and W'
#

etaL = [np.pi/6, np.pi/4, np.pi/3]
chiL = np.linspace(0, np.pi/2, 10)

alphaL = np.linspace(0, 2*np.pi, 10)
betaL = alphaL
thetaL = alphaL

E0_WWitnessed = {True:[], False:[]}
E0_WPrimeWitnessed = {True:[], False:[]}

min_val = [0]*6 
min_state = {}

for eta in etaL:
    for chi in chiL:
        rho_E0 = make_E0(eta, chi)

        for a in a_range:
            b = np.sqrt(1-a**2)
            witnessL = make_W(a,b)
            
            for i in range(len(witnessL)):
                W = witnessL[i]
                valW = (W*(rho_E0)).tr()
                E0_WWitnessed[bool(valW < 0)] += [(a,b,i), (eta, chi)]

                if min_val[i] > valW:
                    min_state[i] =[(a,b), (eta*180/np.pi,chi*180/np.pi)]


        # for alpha in alphaL:
        #     for beta in betaL: 
        #         for theta in thetaL:
        #             wPrimeL = make_WPrimes(alpha, beta, theta)

        #             for i in range(len(wPrimeL)):
        #                 Wp = wPrimeL[i]
        #                 valWp = (Wp*(rho_E0)).tr()
        #                 E0_WPrimeWitnessed[bool(valWp < 0)] += [(alpha, beta, theta, i)] 

print("E0 was witnessed by the Ws ", len(E0_WWitnessed[True]), " times.")
# print("E0 was witnessed by the W's", len(E0_WPrimeWitnessed[True])," times.")
print(E0_WWitnessed[True])

# minimize W_i wrt 
# use compute_witnesses to return the min W, not the entire array (this part just minimizes the a's, b's)









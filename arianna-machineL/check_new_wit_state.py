from qutip import *
from math import sqrt, cos, sin
import numpy as np
from gen_state import make_W_list
from rho_methods_ONLYGOOD import compute_witnesses, get_rho
import matplotlib.pyplot as plt


def pauli_state(W_mat):
    """ Checks which pauli matrix combos make up the witness, W_mat

        0 - IxI 4 - XxI 8 - YxI 12 - ZxI
        1 - IxX 5 - XxX 9 - YxX 13 - ZxX
        2 - IxY 6 - XxY 10 - YxY 14 - ZxY
        3 - IxZ 7 - XxZ 11 - YxZ 15 - ZxZ
    """

    II = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).flatten()
    IX = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]).flatten()
    IY = np.array([[0,-1j,0,0],[1j,0,0,0],[0,0,0,-1j],[0,0,1j,0]]).flatten()
    IZ = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]]).flatten()

    XI = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]).flatten()
    XX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]).flatten()
    XY = np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]]).flatten()
    XZ = np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]]).flatten()

    YI = np.array([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]]).flatten()
    YX = np.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[1j,0,0,0]]).flatten()
    YY = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]).flatten()
    YZ = np.array([[0,0,-1j,0],[0,0,0,1j],[1j,0,0,0],[0,-1j,0,0]]).flatten()

    ZI = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]).flatten()
    ZX = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,-1,0]]).flatten()
    ZY = np.array([[0,-1j,0,0],[1j,0,0,0],[0,0,0,1j],[0,0,-1j,0]]).flatten()
    ZZ = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).flatten()

    W_flat = W_mat.flatten() 
    
    pauli_mats =  [II,IX,IY,IZ,XI,XX,XY,XZ,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ]
    contains_pauli = [False]*16

    for i in range(len(pauli_mats)):
        prod = np.dot(pauli_mats[i],W_flat)
        if prod != 0:
            contains_pauli[i] = True
    return contains_pauli

eta_L = np.array([0,np.pi/4,np.pi/3, np.arccos(-3/5), np.pi/2])
chi = 0


H = basis(2,0) # ket 1
V = basis(2,1) # ket 2
D = 1/sqrt(2) * (H + V)
A = 1/sqrt(2) * (H - V)
R = 1/sqrt(2)*(H + 1j*V)
L = 1/sqrt(2)*(H - 1j*V)

# for i in range(len(eta_L)):
#     eta = eta_L[i]
#     norm_factor = (1 + 0.5*cos(chi)*sin(2*eta))**0.5

#     # states to witness: (normalize them)
#     # HL + LH
#     state_HL = (cos(eta)*tensor(H,L) + sin(eta)*tensor(L,H))/norm_factor
#     # HR + RH
#     state_HR = (cos(eta)*tensor(H,R) + sin(eta)*tensor(R,H))/norm_factor
#     # VL + LV
#     state_VL = (cos(eta)*tensor(V,L) + sin(eta)*tensor(L,V))/norm_factor
#     # VR + RV
#     state_VR = (cos(eta)*tensor(V,R) + sin(eta)*tensor(R,V))/norm_factor

#     state_L = [state_HL,state_HR,state_VL,state_VR]


#     # W states
#     # HL + LH
#     W_state_HL = (tensor(H,L) + tensor(L,H))/norm_factor
#     # HR + RH
#     W_state_HR = (tensor(H,R) + tensor(R,H))/norm_factor
#     # VL + LV
#     W_state_VL = (tensor(V,L) + tensor(L,V))/norm_factor
#     # VR + RV
#     W_state_VR = (tensor(V,R) + tensor(R,V))/norm_factor

#     # HH - LL
#     W_state_HL_s = (tensor(H,H) - tensor(L,L))/norm_factor
#     # HH - RR
#     W_state_HR_s = (tensor(H,H) - tensor(R,R))/norm_factor
#     # VV - LL
#     W_state_VL_s = (tensor(V,V) + tensor(L,L))/norm_factor
#     # VV - RR
#     W_state_VR_s = (tensor(V,V) + tensor(R,R))/norm_factor

#     # HL + i LH
#     W_state_HL_i = (tensor(H,L) + 1j*tensor(L,H))/norm_factor
#     # HR + i RH
#     W_state_HR_i = (tensor(H,R) + 1j*tensor(R,H))/norm_factor
#     # VL + i LV
#     W_state_VL_i = (tensor(V,L) + 1j*tensor(L,V))/norm_factor
#     # VR + i RV
#     W_state_VR_i = (tensor(V,R) + 1j*tensor(R,V))/norm_factor


#     # Generate different witnessses : try opposite state intuition 
#     # W = (phi_ket*phi_bra) partial transpose

#     # (1) HR + RH | HL + LH
#     W_1_HL = partial_transpose(W_state_HR*W_state_HR.dag(), [0,1])
#     W_1_HR = partial_transpose(W_state_HL*W_state_HL.dag(), [0,1])
#     W_1_VL = partial_transpose(W_state_VR*W_state_VR.dag(), [0,1])
#     W_1_VR = partial_transpose(W_state_VL*W_state_VL.dag(), [0,1])
#     W_1_L = [W_1_HL,W_1_HR,W_1_VL,W_1_VR]
    
#     # (2) HR + RH | HH - LL
#     W_2_HL = partial_transpose(W_state_HR_s*W_state_HR_s.dag(), [0,1])
#     W_2_HR = partial_transpose(W_state_HL_s*W_state_HL_s.dag(), [0,1])
#     W_2_VL = partial_transpose(W_state_VR_s*W_state_VR_s.dag(), [0,1])
#     W_2_VR = partial_transpose(W_state_VL_s*W_state_VL_s.dag(), [0,1])
#     W_2_L = [W_2_HL,W_2_HR,W_2_VL,W_2_VR]

#     # (3) HR + RH | HH - RR
#     W_3_HL = partial_transpose(W_state_HL_s*W_state_HL_s.dag(), [0,1])
#     W_3_HR = partial_transpose(W_state_HR_s*W_state_HR_s.dag(), [0,1])
#     W_3_VL = partial_transpose(W_state_VL_s*W_state_VL_s.dag(), [0,1])
#     W_3_VR = partial_transpose(W_state_VR_s*W_state_VR_s.dag(), [0,1])
#     W_3_L = [W_3_HL,W_3_HR,W_3_VL,W_3_VR]

#     # (4) HR + RH | VL + LV
#     W_4_HL = partial_transpose(W_state_VR*W_state_VR.dag(), [0,1])
#     W_4_HR = partial_transpose(W_state_VL*W_state_VL.dag(), [0,1])
#     W_4_VL = partial_transpose(W_state_HR*W_state_HR.dag(), [0,1])
#     W_4_VR = partial_transpose(W_state_HL*W_state_HL.dag(), [0,1])
#     W_4_L = [W_4_HL,W_4_HR,W_4_VL,W_4_VR]
    
#     # (5) flip the imaginary!
#     W_5_HL = partial_transpose(W_state_HL_i*W_state_HL_i.dag(), [0,1])
#     W_5_HR = partial_transpose(W_state_HR_i*W_state_HR_i.dag(), [0,1])
#     W_5_VL = partial_transpose(W_state_VL_i*W_state_VL_i.dag(), [0,1])
#     W_5_VR = partial_transpose(W_state_VR_i*W_state_VR_i.dag(), [0,1])
#     W_5_L = [W_5_HL,W_5_HR,W_5_VL,W_5_VR]

#     for j in range(len(state_L)):
#         W_1_exp = (W_1_L[j]*ket2dm(state_L[j])).tr()
#         W_2_exp = (W_2_L[j]*ket2dm(state_L[j])).tr()
#         W_3_exp = (W_3_L[j]*ket2dm(state_L[j])).tr()
#         W_4_exp = (W_4_L[j]*ket2dm(state_L[j])).tr()
#         W_5_exp = (W_5_L[j]*ket2dm(state_L[j])).tr()
#         # print(f"For eta={eta}, state = {j} \n 1 : {W_1_exp} \n 2 : {W_2_exp} \n 3 : {W_3_exp} \n 4 : {W_4_exp} \n 5 : {W_5_exp} \n ")

# a_L = np.linspace(0,1,100)
a_L = np.linspace(-1,1,10)
alpha_L = np.linspace(0,np.pi,6)
beta_L = np.linspace(0,2*np.pi,6)
theta_L = np.linspace(0,np.pi,6)
phi_L = np.linspace(0,2*np.pi,6)
chi_L = np.linspace(0,2*np.pi,6)
eta_L = np.linspace(0,np.pi,6)
lambda_L = np.linspace(0,2*np.pi,6)
gamma_L = np.linspace(0,np.pi,6)

z_first_corr = False # there needs to be a measurement of zx corr and zy corr
xy_first_corr = False #  there needs to be a measurement of xz corr and yz corr
count = 0

# scan exhastively through all witness possibilities
for alpha in alpha_L:
    for beta in beta_L:
        for theta in theta_L:
            for phi in phi_L:
                for chi in chi_L:
                    for eta in eta_L:
                        for lam in lambda_L:
                            for gamma in gamma_L:

                    # # psi_A chi_B + psip_A chip_B
                    # W_state = tensor(cos(theta)*H + sin(theta)*np.exp(1j*phi)*V, cos(alpha)*H + sin(alpha)*np.exp(1j*beta)*V) + np.exp(1j*chi)*tensor(-np.exp(-1j*phi)*sin(theta)*H + cos(theta)*V,-np.exp(-1j*beta)*sin(alpha)*H + cos(alpha)*V)
                    # W = partial_transpose(W_state*W_state.dag(), [0,1])
                                W_state = tensor(cos(theta)*H + sin(theta)*np.exp(1j*phi)*V, cos(alpha)*H + sin(alpha)*np.exp(1j*beta)*V) + tensor(-np.exp(-1j*chi)*sin(eta)*H + cos(eta)*V,-np.exp(-1j*lam)*sin(gamma)*H + cos(gamma)*V)
                                W = partial_transpose(W_state*W_state.dag(), [0,1])

                                # check if there is an x,z and y,z component simultaneously
                                mat = W.full()
                                
                                pauli_mats = pauli_state(mat)
                                
                                if pauli_mats[7] and pauli_mats[11]:
                                    if not pauli_mats[6] and not pauli_mats[9]:
                                        xy_first_corr = True
                                
                                if pauli_mats[13] and pauli_mats[14]:
                                    if not pauli_mats[6] and not pauli_mats[9]:
                                        z_first_corr = True



                    # define a set of arbitrary HR states to test.
                    # for a in a_L:
                    #     b = np.sqrt(1-a**2)
                    #     test_state = a*tensor(H,V) + b*tensor(V,H)
                    #     rho = ket2dm(test_state)
                    #     exp_val = (W*rho).tr()
                    
                                # test_state = (1/np.sqrt(2))*(tensor(H,V) + tensor(V,H))
                                # rho = ket2dm(test_state)
                                # exp_val = (W*rho).tr()

                                # if np.real(exp_val) < 0 and (z_first_corr or xy_first_corr):
                                if (z_first_corr or xy_first_corr):
                                    print(f"alpha:{alpha}, beta:{beta}, theta:{theta}, phi:{phi}, chi:{chi}")





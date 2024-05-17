from qutip import *
from math import sqrt, cos, sin
import numpy as np
from gen_state import make_W_list
from rho_methods_ONLYGOOD import compute_witnesses, get_rho
import matplotlib.pyplot as plt
# from check_new_wit_state import pauli_state

H = basis(2,0) # ket 1
V = basis(2,1) # ket 2
D = 1/sqrt(2) * (H + V)
A = 1/sqrt(2) * (H - V)
R = 1/sqrt(2)*(H + 1j*V)
L = 1/sqrt(2)*(H - 1j*V)


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

# scan over W'' possibilities and see which measurements are in them 

groups = []

# first scan over ent states "generic witness form" not too complex
alpha_L = np.linspace(0,np.pi,6)
beta_L = np.linspace(0,2*np.pi,6)
theta_L = np.linspace(0,np.pi,6)
phi_L = np.linspace(0,2*np.pi,6)
chi_L = np.linspace(0,2*np.pi,6)
eta_L = np.linspace(0,np.pi,6)
lambda_L = np.linspace(0,2*np.pi,6)
gamma_L = np.linspace(0,np.pi,6)

scan_list = []

for alpha in alpha_L:
    for beta in beta_L:
        for theta in theta_L:
            for phi in phi_L:
                for chi in chi_L:
                    for eta in eta_L:
                        for lam in lambda_L:
                            for gamma in gamma_L:
                                scan_list += [[alpha,beta,theta,phi,chi,eta,lam,gamma]]

def scan_wit(alpha_L, beta_L, theta_L,phi_L,chi_L, eta_L, lambda_L, gamma_L):
    group_count = 0
    groups = {}
    for alpha in alpha_L:
        for beta in beta_L:
            for theta in theta_L:
                for phi in phi_L:
                    for chi in chi_L:
                        for eta in eta_L:
                            for lam in lambda_L:
                                for gamma in gamma_L:
                                    # psi_A chi_B + psip_A chip_B
                                    W_state = tensor(cos(theta)*H + sin(theta)*np.exp(1j*phi)*V, cos(alpha)*H + sin(alpha)*np.exp(1j*beta)*V) + tensor(-np.exp(-1j*chi)*sin(eta)*H + cos(eta)*V,-np.exp(-1j*lam)*sin(gamma)*H + cos(gamma)*V)
                                    W = partial_transpose(W_state*W_state.dag(), [0,1])

                                    # check if there is an x,z and y,z component simultaneously
                                    mat = W.full()
                                    pauli_mats = pauli_state(mat)
                                    # wit_groups = check_groups(pauli_mats) # for checking which groups of measurements are present

                                    groups[group_count] = pauli_mats
                                    group_count += 1
    return groups

def check_groups(mat_list):
    wit_groups = [0, 0, 0, 0]  # [0] is W, [1] is W' 1-3, [2] is W' 4-6, [3] is W' 7-9
    if mat_list[0] or mat_list[1] or mat_list[2] or mat_list[3] or mat_list[4] or mat_list[5] or mat_list[8] or mat_list[10] or mat_list[12] or mat_list[15]:
        wit_groups[0] = 1
    if mat_list[11] or mat_list[14]:
        wit_groups[1] = 1
    if mat_list[7] or mat_list[13]:
        wit_groups[2] = 1
    if mat_list[6] or mat_list[9]:
        wit_groups[3] = 1
    return wit_groups


groups = scan_wit(alpha_L, beta_L, theta_L,phi_L,chi_L,eta_L,lambda_L, gamma_L)
two_groups = {}

for i in range(len(scan_list)):
    if sum(groups[i]) == 3 and groups[i][1] != 0:
        two_groups[i] = groups[i]

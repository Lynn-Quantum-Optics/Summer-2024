# file to hold methods for random generation of density matrices
import numpy as np
import pandas as pd

## note: get_random_jones is in jones.py ##
from rho_methods_ONLYGOOD import *
from sample_rho import *

def get_random_simplex(return_params=False):
    '''
    Returns density matrix for random state of form:
    a|HH> + be^(i*beta)|01> + ce^(i*gamma)*|10> + de^(i*delta)*|11>
    along with generation parameters [a, b, c, d, beta, gamma, delta]
    params:
        return_params: bool, whether to return generation parameters
    '''

    def get_random_state():
    
        a = np.random.rand()
        b = np.random.rand() *(1-a)
        c = np.random.rand()*(1-a-b)
        d = 1-a-b-c

        real_ls = [] # list to store real coefficients
        real_ls = np.sqrt(np.array([a, b, c, d]))
        np.random.shuffle(real_ls)
        rand_angle=np.random.rand(3)*2*np.pi

        params = np.concatenate((real_ls, rand_angle))

        state_vec = np.multiply(real_ls, np.e**(np.concatenate((np.array([1]), rand_angle))*1j)).reshape((4,1))
        # compute density matrix
        rho = get_rho(state_vec)
        return rho, params
    
    rho, params = get_random_state()
    while not(is_valid_rho(rho)):
        rho, params = get_random_state()

    if return_params: return [rho, params]
    else: return rho

def get_random_hurwitz(method=1, log_params=False, conc_cond = 0, purity_cond = 1):
    ''' Function to generate random density matrix with roik method.
    params:
        method: int, 0, 1, 2 for whether to use 
            (0) phi = arcsin(xi^1/n) for n in 1 to 6 with xi : [0,1) or
            (1) phi = arcsin(xi^1/2) for n in 1 to 6 or
            (2) phi = rand in [0, pi/2] for n in 1 to 6
            default is 0.
        log_params: bool, whether to return the parameters used to generate the 3rd unitary matrix
        conc_cond: default is 0, will generate random density matrix until concurrence is >= than this value
        purity_condition: default is 1; will generate random density matrix until purity is less than this value
    '''
    ## part 1: random diagonal elements ##
    def rand_diag():
        # get 4 random params
        
        M11 = np.random.randint(0, 100000) / 100000
        M22 = np.random.randint(0, 100000) / 100000*(1-M11)
        M33 = np.random.randint(0, 100000) / 100000*(1-M11 - M22)
        M44 = 1-M11-M22-M33
        
        # shuffle the entries
        rand_elems = np.array([M11, M22, M33, M44])
        np.random.shuffle(rand_elems)
        M = np.matrix(np.diag([rand_elems[0], rand_elems[1], rand_elems[2], rand_elems[3]]))
        return M

    ## part 2: random unitary trans ##
    global params
    def rand_unitary():
        # need to first generate 6 other smaller unitaries
        if method==1 or method==2:
            def get_rand_elems():
                alpha = np.random.randint(0, 200000*np.pi) / 100000
                psi = np.random.randint(0, 200000*np.pi) / 100000
                chi = np.random.randint(0, 200000*np.pi) / 100000
                if method==1:
                    phi = np.arcsin((np.random.randint(0, 100000)/100000)**(1/2))
                else: # method==2
                    phi = np.random.rand()*np.pi/2
                return np.matrix([
                    [np.e**(psi*1j)*np.cos(phi), np.e**(chi*1j)*np.sin(phi)],
                    [-np.e**(-chi*1j)*np.sin(phi), np.e**(-psi*1j)*np.cos(phi)]
                ])*np.e**(alpha*1j), alpha, psi, chi, phi

            # loop and create unitaries from blocks
            unitary_final = np.eye(4, dtype=np.complex)
            for k in range(5, -1, -1): # count down to do multiplicatiom from right to left
                sub_unitary_k,   alpha, psi, chi, phi = get_rand_elems()
                if k==0 or k==3 or k==5:
                    unitary_k = np.matrix(np.block([[np.eye(2), np.zeros((2,2))], [np.zeros((2,2)), sub_unitary_k]]))
                    if log_params and k==3:
                        global params
                        params = [alpha, psi, chi, phi]
                elif k==1 or k==4:
                    ul = np.matrix([[1,0], [0, sub_unitary_k[0, 0]]])# upper left
                    ur = np.matrix([[0,0], [sub_unitary_k[0, 1], 0]])# upper right
                    ll = np.matrix([[0,sub_unitary_k[1,0]], [0, 0]])# lower left
                    lr = np.matrix([[sub_unitary_k[1,1], 0], [0, 1]])# lower right
                    unitary_k = np.matrix(np.block([[ul, ur],[ll, lr]]))
                else: # k==2
                    unitary_k = np.matrix(np.block([[sub_unitary_k, np.zeros((2,2))], [np.zeros((2,2)), np.eye(2)]]))
                
                # print(np.all(np.isclose(unitary_k@np.linalg.inv(unitary_k), np.eye(4), atol=1e-9)))
                
                unitary_final =  unitary_k @ unitary_final# this way correctly builds right to left
        else: # method==0
            def get_U(i, j, k):
                # get i phi and psi
                phi = np.arcsin((np.random.rand(i+1))**(1/(i+1)))
                psi = np.random.rand(i+1)*2*np.pi
                # get 1 random chi
                if k==0:
                    chi = 0
                else:
                    chi = np.random.rand()*2*np.pi

                # start with identity
                U_k = np.eye(4, dtype=np.complex)
                # fill in the entries
                U_k[i][i]=np.e**(psi[i]*1j)*np.cos(phi[i])
                U_k[i][j]=np.e**(chi*1j)*np.sin(phi[i])
                U_k[j][i]=-np.e**(-chi*1j)*np.sin(phi[i])
                U_k[j][j]=np.e**(-psi[i]*1j)*np.cos(phi[i])

                return U_k

            unitary_final = np.eye(4, dtype=np.complex)
            for l in range(2, -1, -1): # outer loop for the 3 unitaries
                U_l = np.eye(4)
                for k in range(l, -1, -1): # build the unitary from right to left
                    U_l = U_l @ get_U(k, l+1, l)
                unitary_final = unitary_final @ U_l
            
            alpha = np.random.rand()*2*np.pi
            unitary_final = unitary_final * np.e**(alpha*1j)

        return unitary_final

    ## combine M and U as follows: U M U^\dagger
    def combine_rand(): 
        return U @ M @ adjoint(U) # calling adjoint method from rho_methods

    M = rand_diag()
    U = rand_unitary()
    M0 = combine_rand()

    while not(is_valid_rho(M0)) or (get_purity(M0)>purity_cond and get_concurrence(M) >= conc_cond): # if not valid density matrix, keep generating
        print('invalid!!!')
        print(M0)
        M = rand_diag()
        U = rand_unitary()
        M0 = combine_rand()

    # return M0, get_purity(M0)
    if not(log_params): return M0
    else: return [M0, params]

def get_random_werner_simplex():
    ''' Function to generate mixed random density matrix by mixing a random pure state with a maximally mixed one.'''

    def do_mixing():
        ''' Function to mix two density matrices together. '''
        rand_pure = get_random_simplex()
        max_mixed = np.eye(4)/4
        p = np.random.rand()
        return p*rand_pure + (1-p)*max_mixed

    rho = do_mixing()
    while not(is_valid_rho(rho)):
        rho = do_mixing()
    return rho

def get_random_E0(return_params = True):
    ''' Using the defintion of E0 in sample_rho.py
    params:
        return_params: if True, returns the parameters used to generate the state. True by default
    '''

    # get random params
    def get_rand_state():
        eta = np.random.rand()*np.pi/2
        chi = np.random.rand()*2*np.pi

        E0 = get_E0(eta, chi)
        angles = [eta, chi]
        return E0, angles
    E0, angles = get_rand_state()
    while not(is_valid_rho(E0)):
        E0, angles = get_rand_state()

    if return_params: return [E0,angles]
    else: return E0

def get_random_E1(return_params = True):
    ''' Using the defintion of E1 in sample_rho.py'''

    # get random params
    def get_rand_state():
        eta = np.random.rand()*np.pi/2
        chi = np.random.rand()*2*np.pi

        E1 = get_E1(eta, chi)
        angles = [eta, chi]
        return E1, angles
    E1, angles = get_rand_state()
    while not(is_valid_rho(E1)):
        E1, angles = get_rand_state()

    if return_params: return [E1,angles]
    else: return E1


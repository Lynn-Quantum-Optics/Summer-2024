# bigg file perform various calculations on density matrices
# methods adapted my (oscar) code as well as Alec (concurrence)

# main imports #
from os.path import join
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import trange

##############################################
## for more basic stats about a state ##

def get_rho(state):
    ''' Function to compute density matrix from a given 2-qubit state vector. '''
    return np.matrix(state @ np.conjugate(state.reshape((1,4))))

def is_valid_rho(rho, verbose=True):
    ''' Checks if a density matrix is valid. 
    params:
        rho: density matrix to check
        verbose: bool, whether to print out what is wrong with rho
    '''
    # make sure not a 0 matrix
    if np.all(np.isclose(rho, np.zeros((4,4)), rtol=1e-5)):
        if verbose: print('rho is 0 matrix')
        return False
    # check if Hermitian
    if not(np.all(np.isclose(rho,adjoint(rho), rtol=1e-5))):
        if verbose: print('rho is not Hermitian')
        return False
    # check if trace 1, within tolerance. can use param rtol to change tolerance
    if not(np.isclose(np.trace(rho), 1, rtol=1e-5)):
        if verbose: print('rho trace is not 1')
        return False
    # check if positive semidefinite
    if not(np.all(la.eigvals(rho) >= 0)):
        if verbose: print('rho is not positive semidefinite')
        return False
    return True

def adjoint(state):
    ''' Returns the adjoint of a state vector. For a np.matrix, can use .H'''
    return np.conjugate(state).T

def get_purity(rho):
    ''' Calculates the purity of a density matrix. '''
    return np.real(np.trace(rho @ rho))

def get_fidelity(rho1, rho2):
    '''Compute fidelity of 2 density matrices'''
    return (np.real(np.trace(la.sqrtm(la.sqrtm(rho1)@rho2@la.sqrtm(rho1)))))**2

def Bures_distance(rho1, rho2):
    '''Compute the distance between 2 density matrices'''
    fidelity = get_fidelity(rho1, rho2)
    return np.sqrt(2*(1-np.sqrt(fidelity)))

##############################################
## for tomography ##

def get_expec_vals(rho):
    ''' Returns all 16 expectation vals given density matrix. '''
    # pauli matrices
    I = np.eye(2)
    X = np.matrix([[0, 1], [1, 0]])
    Y = np.matrix([[0, -1j], [1j, 0]])
    Z = np.matrix([[1, 0], [0, -1]])

    expec_vals=[]

    Pa_matrices = [I, X, Y, Z]
    for Pa1 in Pa_matrices:
        for Pa2 in Pa_matrices:
            expec_vals.append(np.trace(np.kron(Pa1, Pa2) @ rho))

    return np.array(expec_vals).reshape(4,4)

def compute_proj(basis1, basis2, rho):
    ''' Computes projection into desired bases using projection operations on both qubits'''
    # get projection operators
    proj1 = basis1 @ adjoint(basis1)
    proj2 = basis2 @ adjoint(basis2)

    # print(np.kron(proj1, proj2))
    return np.real(np.trace(np.kron(proj1, proj2)@rho))

def get_9s_projections(rho):
    ''' Computes 9 projections based on the standard projection'''
    ## define the single qubit bases for projection: still want our input vectors to be these probabilities ##
    H = np.array([[1,0]]).reshape(2,1) # ZP
    V = np.array([[0,1]]).reshape(2,1) # ZM
    D = 1/np.sqrt(2) * np.array([[1,1]]).reshape(2,1) # XP
    A = 1/np.sqrt(2) * np.array([[1,-1]]).reshape(2,1) # XM
    R = 1/np.sqrt(2) * np.array([[1,1j]]).reshape(2,1) # YP
    L = 1/np.sqrt(2) * np.array([[1,-1j]]).reshape(2,1) # YM

    HH = compute_proj(H, H, rho)
    HV = compute_proj(H, V, rho)
    # VH = compute_proj(V, H, rho)
    VV = compute_proj(V, V, rho)
    DD = compute_proj(D, D, rho)
    DA = compute_proj(D, A, rho)
    # AD = compute_proj(A, D, rho)
    AA = compute_proj(A, A, rho)
    RR = compute_proj(R, R, rho)
    RL = compute_proj(R, L, rho)
    # LR = compute_proj(L, R, rho)
    LL = compute_proj(L, L, rho)

    # return HH, HV, VH, VV, DD, DA, AD, AA, RR, RL, LR, LL
    return HH, HV, VV, DD, DA, AA, RR, RL, LL

def get_12s_redundant_projections(rho):
    ''' Computes 12 projections based on the standard projection'''

     ## define the single qubit bases for projection: still want our input vectors to be these probabilities ##
    H = np.array([[1,0]]).reshape(2,1) # ZP
    V = np.array([[0,1]]).reshape(2,1) # ZM
    D = 1/np.sqrt(2) * np.array([[1,1]]).reshape(2,1) # XP
    A = 1/np.sqrt(2) * np.array([[1,-1]]).reshape(2,1) # XM
    R = 1/np.sqrt(2) * np.array([[1,1j]]).reshape(2,1) # YP
    L = 1/np.sqrt(2) * np.array([[1,-1j]]).reshape(2,1) # YM

    HH = compute_proj(H, H, rho)
    HV = compute_proj(H, V, rho)
    VH = compute_proj(V, H, rho)
    VV = compute_proj(V, V, rho)
    DD = compute_proj(D, D, rho)
    DA = compute_proj(D, A, rho)
    AD = compute_proj(A, D, rho)
    AA = compute_proj(A, A, rho)
    RR = compute_proj(R, R, rho)
    RL = compute_proj(R, L, rho)
    LR = compute_proj(L, R, rho)
    LL = compute_proj(L, L, rho)

    return HH, HV, VH, VV, DD, DA, AD, AA, RR, RL, LR, LL
    

def get_16s_projections(rho):
    ''' Computes 16 projections based on the standard projection; choice of bases is motivated by Roik et al'''
    ## define the single qubit bases for projection: still want our input vectors to be these probabilities ##
    H = np.array([[1,0]]).reshape(2,1) # ZP
    V = np.array([[0,1]]).reshape(2,1) # ZM
    D = 1/np.sqrt(2) * np.array([[1,1]]).reshape(2,1) # XP
    A = 1/np.sqrt(2) * np.array([[1,-1]]).reshape(2,1) # XM
    R = 1/np.sqrt(2) * np.array([[1,1j]]).reshape(2,1) # YP
    L = 1/np.sqrt(2) * np.array([[1,-1j]]).reshape(2,1) # YM

    HH = compute_proj(H, H, rho)
    VV = compute_proj(V, V, rho)
    HV = compute_proj(H, V, rho)

    DD = compute_proj(D, D,rho)
    AA = compute_proj(A, A,rho)

    RR = compute_proj(R,R, rho)
    LL = compute_proj(L, L,rho)

    DL = compute_proj(D,L, rho)
    AR = compute_proj(A,R, rho)
    DH = compute_proj(D,H, rho)
    AV = compute_proj(A,V, rho)
    LH = compute_proj(L,H, rho)
    RV = compute_proj(R,V, rho)

    DR = compute_proj(D,R, rho)
    DV = compute_proj(D,V, rho)
    LV = compute_proj(L,V, rho)

    return HH, VV, HV, DD, AA, RR, LL, DL, AR, DH, AV, LH, RV, DR, DV, LV


def compute_roik_proj(basis1, basis2, M0):
        ''' Computes projection into desired bases as in the Roik et al paper'''
        Bell_singlet = np.matrix([[0, 0, 0, 0], [0, .5, -.5, 0], [0, -0.5, .5, 0], [0, 0, 0, 0]])

        M0_swapped = M0.copy() # swap the subsystems A and B
        M0_swapped[:, 1] = M0[:, 2]
        M0_swapped[:, 2] = M0[:, 1]

        M_T = np.kron(M0, M0_swapped)
        num = M_T @ np.kron(np.kron(basis1, Bell_singlet), basis2)
        denom = M_T @ np.kron(np.kron(basis1, np.eye(4)), basis2)

        try: # compute the projection as defined in Roik et al
            return (np.trace(num) / np.trace(denom)).real
        except ZeroDivisionError:
            return 0

def get_all_roik_projections(M0):
    ''' Computes the 16 projections as defined in Roik et al'''

    # define the single bases for projection
    H = np.array([[1,0],[0,0]])
    V = np.array([[0,0],[0,1]])    
    D = np.array([[1/2,1/2],[1/2,1/2]])
    A = np.array([[1/2,-1/2],[-1/2,1/2]])
    R = np.array([[1/2,1j/2],[-1j/2,1/2]])
    L = np.array([[1/2,-1j/2],[1j/2,1/2]])

    HH = compute_roik_proj(H, H, M0)
    VV = compute_roik_proj(V, V, M0)
    HV = compute_roik_proj(H, V, M0)

    DD = compute_roik_proj(D, D, M0)
    AA = compute_roik_proj(A, A, M0)

    RR = compute_roik_proj(R,R, M0)
    LL = compute_roik_proj(L, L, M0)

    DL = compute_roik_proj(D,L, M0)
    AR = compute_roik_proj(A,R, M0)
    DH = compute_roik_proj(D,H, M0)
    AV = compute_roik_proj(A,V, M0)
    LH = compute_roik_proj(L,H, M0)
    RV = compute_roik_proj(R,V, M0)

    DR = compute_roik_proj(D,R, M0)
    DV = compute_roik_proj(D,V, M0)
    LV = compute_roik_proj(L,V, M0)

    return HH, VV, HV, DD, AA, RR, LL, DL, AR, DH, AV, LH, RV, DR, DV, LV

def compute_witnesses(rho):
    ''' Computes the minimum of the 6 Ws and the minimum of the 3 triples of the 9 W's. '''
    expec_vals = get_expec_vals(rho)

    def get_W1(theta, expec_vals):
        a, b = np.cos(theta), np.sin(theta)
        return np.real(0.25*(expec_vals[0,0] + expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] + (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[3,0] + expec_vals[0,3])))
    def get_W2(theta, expec_vals):
        a, b = np.cos(theta), np.sin(theta)
        return np.real(0.25*(expec_vals[0,0] - expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] - (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[3,0] - expec_vals[0,3])))
    def get_W3(theta, expec_vals):
        a, b = np.cos(theta), np.sin(theta)
        return np.real(0.25*(expec_vals[0,0] + expec_vals[1,1] + (a**2 - b**2)*expec_vals[3,3] + (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[1,0] + expec_vals[0,1])))
    def get_W4(theta, expec_vals):
        a, b = np.cos(theta), np.sin(theta)
        return np.real(0.25*(expec_vals[0,0] - expec_vals[1,1] + (a**2 - b**2)*expec_vals[3,3] - (a**2 - b**2)*expec_vals[2,2] - 2*a*b*(expec_vals[1,0] - expec_vals[0,1])))
    def get_W5(theta, expec_vals):
        a, b = np.cos(theta), np.sin(theta)
        return np.real(0.25*(expec_vals[0,0] + expec_vals[2,2] + (a**2 - b**2)*expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] + 2*a*b*(expec_vals[2,0] + expec_vals[0,2])))
    def get_W6(theta, expec_vals):
        a, b = np.cos(theta), np.sin(theta)
        return np.real(0.25*(expec_vals[0,0] - expec_vals[2,2] + (a**2 - b**2)*expec_vals[3,3] - (a**2 - b**2)*expec_vals[1,1] - 2*a*b*(expec_vals[2,0] - expec_vals[0,2])))
    
    ## W' from summer 2022 ##
    def get_Wp1(params, expec_vals):
        theta, alpha = params[0], params[1]
        return np.real(.25*(expec_vals[0,0] + expec_vals[3,3] + np.cos(2*theta)*(expec_vals[1,1]+expec_vals[2,2])+np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.sin(alpha)*(expec_vals[1,2] - expec_vals[2,1])))
    def get_Wp2(params, expec_vals):
        theta, alpha = params[0], params[1]
        return np.real(.25*(expec_vals[0,0] - expec_vals[3,3] + np.cos(2*theta)*(expec_vals[1,1]-expec_vals[2,2])+np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,0] - expec_vals[0,3]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[1,2] - expec_vals[2,1])))
    def get_Wp3(params, expec_vals):
        theta, alpha, beta = params[0], params[1], params[2]
        return np.real(.25 * (np.cos(theta)**2*(expec_vals[0,0] + expec_vals[3,3]) + np.sin(theta)**2*(expec_vals[0,0] - expec_vals[3,3]) + np.cos(theta)**2*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(theta)**2*np.cos(2*alpha - beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*expec_vals[1,0] + np.sin(2*theta)*np.cos(alpha - beta)*expec_vals[0,1] + np.sin(2*theta)*np.sin(alpha)*expec_vals[2,0] + np.sin(2*theta)*np.sin(alpha - beta)*expec_vals[0,2]+np.cos(theta)**2*np.sin(beta)*(expec_vals[2,1] - expec_vals[1,2]) + np.sin(theta)**2*np.sin(2*alpha - beta)*(expec_vals[2,1] + expec_vals[1,2])))
    def get_Wp4(params, expec_vals):
        theta, alpha = params[0], params[1]
        return np.real(.25*(expec_vals[0,0]+expec_vals[1,1]+np.cos(2*theta)*(expec_vals[3,3] + expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[0,1] + expec_vals[1,0]) + np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,3] - expec_vals[3,2])))
    def get_Wp5(params, expec_vals):
        theta, alpha = params[0], params[1]
        return np.real(.25*(expec_vals[0,0]-expec_vals[1,1]+np.cos(2*theta)*(expec_vals[3,3] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[0,1] - expec_vals[1,0]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,3] - expec_vals[3,2])))
    def get_Wp6(params,expec_vals):
        theta, alpha, beta = params[0], params[1], params[2]
        return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(expec_vals[0,0] + expec_vals[3,3] + expec_vals[3,0] + expec_vals[0,3]) + np.cos(theta)**2*np.sin(alpha)**2*(expec_vals[0,0] - expec_vals[3,3] + expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.cos(beta)**2*(expec_vals[0,0] + expec_vals[3,3] - expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.sin(beta)**2*(expec_vals[0,0] - expec_vals[3,3] - expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(expec_vals[2,3] + expec_vals[2,0]) + np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(expec_vals[2,3] - expec_vals[2,0]) - np.cos(theta)**2*np.sin(2*alpha)*(expec_vals[3,2] + expec_vals[0,2]) - np.sin(theta)**2*np.sin(2*beta)*(expec_vals[3,2] - expec_vals[0,2])))
    def get_Wp7(params, expec_vals):
        theta, alpha = params[0], params[1]
        return np.real(.25*(expec_vals[0,0] + expec_vals[2,2]+np.cos(2*theta)*(expec_vals[3,3] + expec_vals[1,1]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,1] - expec_vals[1,3]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,0]+expec_vals[0,2])))
    def get_Wp8(params, expec_vals):
        theta, alpha = params[0], params[1]
        return np.real(.25*(expec_vals[0,0] - expec_vals[2,2] + np.cos(2*theta)*(expec_vals[3,3]-expec_vals[1,1]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,1]+expec_vals[1,3])+np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,0] - expec_vals[0,2])))
    def get_Wp9(params, expec_vals):
        theta, alpha, beta = params[0], params[1], params[2]
        return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(expec_vals[0,0] + expec_vals[3,3] + expec_vals[3,0] + expec_vals[0,3]) + np.cos(theta)**2*np.sin(alpha)**2*(expec_vals[0,0] - expec_vals[3,3] + expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.cos(beta)**2*(expec_vals[0,0] + expec_vals[3,3] - expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.sin(beta)**2*(expec_vals[0,0] - expec_vals[3,3] - expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.cos(theta)**2*np.sin(2*alpha)*(expec_vals[0,1] + expec_vals[3,1]) + np.sin(theta)**2*np.sin(2*beta)*(expec_vals[0,1] - expec_vals[3,1]) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(expec_vals[1,0] + expec_vals[1,3])+ np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(expec_vals[1,0] - expec_vals[1,3])))

    # print('rho', rho)
    # print('expec vals', expec_vals)
    # print('testing W', get_W2(np.pi, expec_vals))

    # now perform optimization; break into three groups based on the number of params to optimize
    all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
    W_expec_vals = []
    for i, W in enumerate(all_W):
        if i <= 5: # just theta optimization
            W_expec_vals.append(minimize(W, x0=[0], args = (expec_vals,), bounds=[(0, np.pi/2)])['fun']) # gave an error with [0] after ['fun']
        elif i==8 or i==11 or i==14: # theta, alpha, and beta
                W_expec_vals.append(minimize(W, x0=[0, 0, 0], args = (expec_vals,), bounds=[(0, np.pi/2), (0, 2*np.pi), (0, 2*np.pi)])['fun'])
        else:# theta and alpha
            W_expec_vals.append(minimize(W, x0=[0, 0], args = (expec_vals,), bounds=[(0, np.pi/2), (0, 2*np.pi)])['fun'])
    
    # find min W expec value; this tells us if first 12 measurements are enough #
    try:
        W_min = np.real(min(W_expec_vals[:6]))[0] ## for some reason, on python 3.9.7 this is a list of length 1, so need to index into it. on 3.10.6 it's just a float 
    except TypeError: # if it's a float, then just use that
        W_min = np.real(min(W_expec_vals[:6]))

    Wp_t1 = np.real(min(W_expec_vals[6:9]))
    Wp_t2 = np.real(min(W_expec_vals[9:12]))
    Wp_t3 = np.real(min(W_expec_vals[12:15]))

    return W_min, Wp_t1, Wp_t2, Wp_t3


##############################################
## for entanglement verification ##

def get_min_eig(M0):
    '''
    Computes the eigenvalues of the partial transpose; if at least one is negative, then state labeled as '0' for entangled; else, '1'. 
    '''
    def partial_transpose():
        # decompose M0 into blocks
        b1 = M0[:2, :2]
        b2 = M0[:2, 2:]
        b3 = M0[2:, :2]
        b4 = M0[2:, 2:]

        PT = np.matrix(np.block([[b1.T, b2.T], [b3.T, b4.T]]))
        return PT

    # compute partial tranpose
    PT = partial_transpose()
    eigenvals = la.eigvals(PT)
    eigenvals.sort() # sort

    return np.real(eigenvals[0]) # return min eigenvalue

def get_concurrence(rho):
    ''' Calculates concurrence of a density matrix using R matrix. '''
    def R_matrix(rho):
        ''' Calculates the Hermitian R matrix for finding concurrence. '''
        def spin_flip(rho):
            ''' Returns the 'spin-flipped' (tilde) version of a density matrix rho.'''
            # define spin operators
            sy = np.array([[0,-1j],[1j,0]])
            sysy = np.kron(sy,sy)
            # perform spin flipping
            return sysy @ rho.conj() @ sysy
        sqrt_rho = la.sqrtm(rho)
        rho_tilde = spin_flip(rho)
        return (sqrt_rho @ rho_tilde @ sqrt_rho)
    R = R_matrix(rho)
    eig_vals = np.real(la.eigvals(R))
    eig_vals = np.sort(eig_vals)[::-1] # reverse sort numpy array
    return np.max([0,eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3]])

def check_conc_min_eig(rho, printf=False):
    ''' Returns both concurence and min eigenvalue of partial transpose. '''
    concurrence = get_concurrence(rho)
    min_eig = get_min_eig(rho)
    if printf:
        print('Concurrence: ', concurrence)
        print('Min eigenvalue: ', min_eig)
    return concurrence, min_eig




##############################################
## for testing ##
if __name__ == '__main__':
    ## testing randomization processes ##
    # random state gen imports #
    from jones import get_random_jones
    from random_gen import get_random_simplex, get_random_roik

    def check_conc_min_eig_sample(N=10000, conditions=None, func=get_random_simplex, method_name='Simplex', savedir='rho_test_plots', special_name='0', display=False, fit=False):
        ''' Checks random sample of N simplex generated matrices. 
        params:
            N: number of random states to check
            conditions: list of conditions to check for: tuple of tuple for min max bounds for concurrence and min eigenvalue. e.g. ((0, .5), (0, -.5)) will ensure states have concurrence between 0 and 0.5 and min eigenvalue between 0 and -0.5
            func: function to generate random state
            method_name: name of method used to generate random state
            savedir: directory to save plots
            special_name: if searching with specific conditions, add a unique name to the plot
            display: whether to display plot
            fit: whether to fit a func to the data
        '''
        concurrence_ls = []
        min_eig_ls = []
        for n in trange(N):
        # for n in range(N):
            # get state
            def get_state():
                if func.__name__ == 'get_random_jones' or func.__name__ == 'get_random_simplex':
                    rho = func()[0]
                else:
                    rho=func()
                return rho
            # impose conditions
            if conditions != None:
                go=False
                while not(go):
                    rho = get_state()
                    concurrence, min_eig = check_conc_min_eig(rho)
                    if conditions[0][0] <= concurrence <= conditions[0][1] and conditions[1][0] <= min_eig <= conditions[1][1]:
                        # print(is_valid_rho(state))
                        go=True
                    else:
                        concurrence, min_eig = check_conc_min_eig(get_state())
            else:
                # check if entangled
                concurrence, min_eig = check_conc_min_eig(get_state())
            # plot
            concurrence_ls.append(concurrence)
            min_eig_ls.append(min_eig)

        # fig, axes = plt.subplots(2,1, figsize=(10,5))
        # axes[0].hist(concurrence_ls, bins=100)
        # axes[1].hist(min_eig_ls, bins=100)
        # plt.show()
        plt.figure(figsize=(10,7))
        plt.plot(concurrence_ls, min_eig_ls, 'o', label='Random states')
        plt.xlabel('Concurrence')
        plt.ylabel('Min eigenvalue')
        plt.title('Concurrence vs. PT Min Eigenvalue for %s'%method_name)
        
        if fit: # fit exponential!
            from scipy.optimize import curve_fit
            def func(x, a, b,c, d, e, f, g):
                # return a*np.exp(-b*(x+c)) + d*x**3 + e*x**2 +f*x+g
                return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6
            popt, pcov = curve_fit(func, concurrence_ls, min_eig_ls)
            perr = np.sqrt(np.diag(pcov))
            conc_lin = np.linspace(min(concurrence_ls), max(concurrence_ls), 1000)

            # calculate chi2red
            # chi2red= np.sum((np.array(min_eig_ls) - func(np.array(concurrence_ls), *popt))**2)/(len(min_eig_ls) - len(pcov))
            # plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-')
            # plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-', label='$\lambda_{min}= %5.3f e^{-%5.3f (x+%5.3f)}+ %5.3fx^3 + %5.3fx^2 + %5.3fx + %5.3f \pm (%5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f), \chi^2_\\nu = %5.3f$'%(*popt, *perr, chi2red))
            # plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-', label='$\lambda_{min}= %5.3f e^{-%5.3f (x+%5.3f)} %5.3fx^3  +%5.3fx^2  %5.3fx  %5.3f$'%(*popt,))
            plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-', label='$\lambda_{min}= %5.3f + %5.3fx + %5.3fx^2 + %5.3fx^3 + %5.3fx^4 + %5.3fx^5 + %5.3fx^6$'%(*popt,))
            plt.legend()

        plt.savefig('%s/conc_min_eig_%s_%s.pdf'%(savedir, method_name, special_name))

        if display:
            plt.show()

    # no conditions
    # check_conc_min_eig_sample(fit=True, method_name='Simplex', func=get_random_simplex, special_name='fit')
    check_conc_min_eig_sample(fit=True, method_name='Roik', func=get_random_roik, special_name='fit')
    # check_conc_min_eig_sample(fit=True, method_name='Jones', func=get_random_jones, special_name='fit')
    # check_conc_min_eig_sample(func=get_random_roik, method_name='roik')
    # check_conc_min_eig_sample(func=get_random_jones, method_name='jones')

    # conditions:
        # investigating typeI and type2 errors: type1 = concurrence = 0, min_eig < 0; type2 = concurrence > 0, min_eig > 0
    # check_conc_min_eig_sample(N=100, method_name='jones', conditions=((0, 0), (-1000, 0)), func=get_random_jones, special_name='type1')
    # check_conc_min_eig_sample(N=1000, method_name='roik', conditions=((0, 0), (-1000, 0)), func=get_random_roik, special_name='conc_0')
    pass
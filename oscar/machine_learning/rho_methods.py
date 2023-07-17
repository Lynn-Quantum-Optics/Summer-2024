# bigg file perform various calculations on density matrices
# methods adapted my (oscar) code as well as Alec (concurrence)

# main imports #
from os.path import join
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime
from tqdm import trange

from uncertainties import ufloat
from uncertainties import unumpy as unp

##############################################
## for more basic stats about a state ##

def get_rho(state):
    ''' Function to compute density matrix from a given 2-qubit state vector. '''
    return np.matrix(state @ np.conjugate(state.reshape((1,4))))

def adjoint(state):
    ''' Returns the adjoint of a state vector. For a np.matrix, can use .H'''
    return np.conjugate(state).T

def is_valid_rho(rho, verbose=True):
    ''' Checks if a density matrix is valid. 
    params:
        rho: density matrix to check
        verbose: bool, whether to print out what is wrong with rho
    '''
    tolerance = 1e-17
    # make sure not a 0 matrix
    if np.all(np.isclose(rho, np.zeros((4,4)), rtol=tolerance)):
        if verbose: print('rho is 0 matrix')
        return False
    # check if Hermitian
    if not(np.all(np.isclose(rho,adjoint(rho), rtol=tolerance))):
        if verbose: print('rho is not Hermitian')
        return False
    # check if trace 1, within tolerance. can use param rtol to change tolerance
    if not(np.isclose(np.trace(rho), 1, tolerance)):
        if verbose: print('rho trace is not 1', np.trace(rho))
        return False
    # check if positive semidefinite
    eig_val = la.eigvals(rho)
    if not(np.all(np.greater_equal(eig_val,np.zeros(len(eig_val))) | np.isclose(eig_val,np.zeros(len(eig_val)), rtol=tolerance))):
        if verbose: print('rho is not positive semidefinite. eigenvalues:', eig_val)
        return False
    # square root of rho must exist
    if np.isnan(rho).any() or np.isinf(rho).any():
        if verbose: 
            print('rho has infs or nans')
            print('nan', np.isnan(rho))
            print('inf', np.isinf(rho))
            print(rho)
        return False
    return True

def get_purity(rho):
    ''' Calculates the purity of a density matrix. '''
    return np.real(np.trace(rho @ rho))

def get_fidelity(rho1, rho2):
    '''Compute fidelity of 2 density matrices'''
    return np.real((np.trace(la.sqrtm(la.sqrtm(rho1)@rho2@la.sqrtm(rho1))))**2)

def Bures_distance(rho1, rho2):
    '''Compute the distance between 2 density matrices'''
    fidelity = get_fidelity(rho1, rho2)
    return np.sqrt(2*(1-np.sqrt(fidelity)))

##############################################
## for tomography ##

def get_expec_vals(rho):
    ''' Returns all 16 expectation vals given density matrix. '''
    # pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.matrix([[0, 1], [1, 0]], dtype=complex)
    Y = np.matrix([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.matrix([[1, 0], [0, -1]], dtype=complex)

    expec_vals=[]

    Pa_matrices = [I, X, Y, Z]
    for Pa1 in Pa_matrices:
        for Pa2 in Pa_matrices:
            expec_vals.append(np.trace(np.kron(Pa1, Pa2) @ rho))

    return np.array(expec_vals).reshape(4,4)

def get_expec_vals_counts(raw_data):
    '''Takes in unumpy array with counts and count uncertainities. Returns expectation values and uncertainties.'''
    # normalize groups of orthonormal measurements to get projections
    projs = np.zeros_like(raw_data)
    for i in range(0,6,2):
        for j in range(0,6,2):
            total_rate = np.sum(raw_data[i:i+2, j:j+2])
            projs[i:i+2, j:j+2] = raw_data[i:i+2, j:j+2]/total_rate
    
    
    HH, HV, HD, HA, HR, HL = projs[0]
    VH, VV, VD, VA, VR, VL = projs[1]
    DH, DV, DD, DA, DR, DL = projs[2]
    AH, AV, AD, AA, AR, AL = projs[3]
    RH, RV, RD, RA, RR, RL = projs[4]
    LH, LV, LD, LA, LR, LL = projs[5]

    # build the stokes's parameters
    S = np.zeros((4,4), dtype=object)
    S[0,0] = 1
    S[0,1] = DD - DA + AD - AA
    S[0,2] = RR + LR - RL - LL
    S[0,3] = HH - HV + VH - VV
    S[1,0] = DD + DA - AD - AA
    S[1,1] = DD - DA - AD + AA
    S[1,2] = DR - DL - AR + AL
    S[1,3] = DH - DV - AH + AV
    S[2,0] = RR - LR + RL - LL
    S[2,1] = RD - RA - LD + LA
    S[2,2] = RR - RL - LR + LL
    S[2,3] = RH - RV - LH + LV
    S[3,0] = HH + HV - VH - VV
    S[3,1] = HD - HA - VD + VA
    S[3,2] = HR - HL - VR + VL
    S[3,3] = HH - HV - VH + VV

    return S
    

def compute_proj(basis1, basis2, rho):
    ''' Computes projection into desired bases using projection operations on both qubits'''
    # get projection operators
    proj1 = basis1 @ adjoint(basis1)
    proj2 = basis2 @ adjoint(basis2)
    # compute projection
    return np.real(np.trace(np.kron(proj1, proj2)@rho))

def get_all_projs(rho):
    ''' Computes all 36 projections for a given density matrix rho'''

    # define the single bases for projection
   # -- old -- #
    # H = np.array([[1,0],[0,0]]) 
    # V = np.array([[0,0],[0,1]])    
    # D = np.array([[1/2,1/2],[1/2,1/2]])
    # A = np.array([[1/2,-1/2],[-1/2,1/2]])
    # R = np.array([[1/2,-1j/2],[1j/2,1/2]])
    # L = np.array([[1/2,1j/2],[-1j/2,1/2]])
    # ------- #
    H = np.array([[1], [0]])
    V = np.array([[0], [1]])
    D = np.array([[1], [1]])/np.sqrt(2)
    A = np.array([[1], [-1]])/np.sqrt(2)
    R = np.array([[1], [1j]])/np.sqrt(2)
    L = np.array([[1], [-1j]])/np.sqrt(2)

    basis_ls =[H, V, D, A, R, L]
    all_projs =[]
    for basis in basis_ls:
        for basis2 in basis_ls:
            all_projs.append(compute_proj(basis, basis2, rho))

    return np.array(all_projs).reshape(6,6)

def reconstruct_rho(all_projs):
    ''' Takes in all 36 projections and reconstructs the density matrix. Based on Beili Hu's thesis.'''
    # unpack the projections
    HH, HV, HD, HA, HR, HL = all_projs[0]
    VH, VV, VD, VA, VR, VL = all_projs[1]
    DH, DV, DD, DA, DR, DL = all_projs[2]
    AH, AV, AD, AA, AR, AL = all_projs[3]
    RH, RV, RD, RA, RR, RL = all_projs[4]
    LH, LV, LD, LA, LR, LL = all_projs[5]
    
    # build the stokes's parameters
    S = np.zeros((4,4))
    S[0,0]=1
    S[0,1] = DD - DA + AD - AA
    S[0,2] = RR + LR - RL - LL
    S[0,3] = HH - HV + VH - VV
    S[1,0] = DD + DA - AD - AA
    S[1,1] = DD - DA - AD + AA
    S[1,2] = DR - DL - AR + AL
    S[1,3] = DH - DV - AH + AV
    S[2,0] = RR - LR + RL - LL
    S[2,1] = RD - RA - LD + LA
    S[2,2] = RR - RL - LR + LL
    S[2,3] = RH - RV - LH + LV
    S[3,0] = HH + HV - VH - VV
    S[3,1] = HD - HA - VD + VA
    S[3,2] = HR - HL - VR + VL
    S[3,3] = HH - HV - VH + VV


    # define pauli matrices
    I = np.eye(2)
    X = np.matrix([[0, 1], [1, 0]])
    Y = np.matrix([[0, -1j], [1j, 0]])
    Z = np.matrix([[1, 0], [0, -1]])
    P = [I, X, Y, Z]

    # compute rho
    rho = np.zeros((4,4), dtype=complex)
    for i1 in range(4):
        for i2 in range(4):
            rho+= S[i1,i2]*np.kron(P[i1],P[i2])

    return rho/4 # scale by 4 to get the correct density matrix

def test_reconstruct_rho(rho):
    ''' Test the reconstruction of the density matrix using the stokes parameters calculated directly from the density matrix.'''
    S = get_expec_vals(rho)
    # define pauli matrices
    I = np.eye(2)
    X = np.matrix([[0, 1], [1, 0]])
    Y = np.matrix([[0, -1j], [1j, 0]])
    Z = np.matrix([[1, 0], [0, -1]])
    P = [I, X, Y, Z]

    # compute rho
    rho = np.zeros((4,4), dtype=complex)
    for i1 in range(4):
        for i2 in range(4):
            rho+= S[i1,i2]*np.kron(P[i1],P[i2])

    return rho/4 # scale by 4 to get the correct density matrix

def compute_roik_proj(basis1, basis2, rho):
    ''' Computes projection into desired bases as in the Roik et al paper'''
    Bell_singlet = np.matrix([[0, 0, 0, 0], [0, .5, -.5, 0], [0, -0.5, .5, 0], [0, 0, 0, 0]])

    rho_swapped = rho.copy() # swap the subsystems A and B
    rho_swapped[:, 1] = rho[:, 2]
    rho_swapped[:, 2] = rho[:, 1]

    M_T = np.kron(rho, rho_swapped)
    num = M_T @ np.kron(np.kron(basis1, Bell_singlet), basis2)
    denom = M_T @ np.kron(np.kron(basis1, np.eye(4)), basis2)

    try: # compute the projection as defined in Roik et al
        return (np.trace(num) / np.trace(denom)).real
    except ZeroDivisionError:
        return 0

def get_all_roik_projs(rho):
    ''' Computes the projections as defined in Roik et al'''

    # define the single bases for projection
    H = np.array([[1], [0]])
    V = np.array([[0], [1]])
    D = np.array([[1], [1]])/np.sqrt(2)
    A = np.array([[1], [-1]])/np.sqrt(2)
    R = np.array([[1], [1j]])/np.sqrt(2)
    L = np.array([[1], [-1j]])/np.sqrt(2)

    basis_ls =[H, V, D, A, R, L]
    all_projs =[]
    for basis in basis_ls:
        for basis2 in basis_ls:
            all_projs.append(compute_roik_proj(basis, basis2, rho))

    return np.array(all_projs).reshape(6,6)

    
def adjust_rho(rho, angles, expt_purity, state='E0'):
    ''' Adjusts density matrix to account for experimental impurity.'''
    if state=='E0':
        r_hv = (1 + np.cos(np.deg2rad(angles[1]))*np.sin(2*np.deg2rad(angles[0]))) / 2
        r_vh = 1 - r_hv
        HV= np.array([0, 1, 0, 0]).reshape(4,1)
        VH = np.array([0, 0, 1, 0]).reshape(4,1)
        rho_adj = expt_purity * rho + (1 - expt_purity) * (r_hv * HV @ adjoint(HV) + r_vh * VH @ adjoint(VH))
        return rho_adj

def get_adj_fidelity(rho, angles, expt_purity, state='E0'):
    ''' Computes the fidelity of the adjusted density matrix with the theoretical density matrix.'''
    adj_rho = adjust_rho(rho, angles, expt_purity, state=state)
    return get_fidelity(adj_rho, rho)



def compute_witnesses(rho, counts = None, expt = False, do_stokes=False, do_counts = False, calc_unc=False, stokes_unc = None, expt_purity = None, angles = None, num_reps = 20, optimize = True, gd=True, zeta=0.7, ads_test=False):
    ''' Computes the minimum of the 6 Ws and the minimum of the 3 triples of the 9 W's. 
        Params:
            rho: the density matrix
            counts: raw unp array of counts and unc
            expt: bool, whether to compute the Ws assuming input is experimental data
            do_stokes: bool, whether to compute 
            calc_unc: bool, whether to compute the uncertainty in the Ws; uses the Stokes params -- must have do_stokes=True. NOTE: THIS OPTION IS DEPRICATED BECAUSE USING THE STOKES UNCERTAINTIES DOESNT PROPPAGATE THE UNCERTAINTIES IN THE Ws CORRECTLY. INSTEAD, PROVIDE A UNUMPY ARRAY OF COUNTS AND UNCERTAINTIES AND PASS EXPT=TRUE.
            stokes_unc: the uncertainty in the stokes params; 4x4 matrix
            do_counts: use the raw definition in terms of counts
            expt_purity: the experimental purity of the state, which defines the noise level: 1 - purity.
            angles: angles of eta, chi for E0 states to adjust theory
            num_reps: int, number of times to run the optimization
            optimize: bool, whether to optimize the Ws with random or gradient descent or to just check bounds
            gd: bool, whether to use gradient descent or brute random search
            zeta: learning rate for gradient descent
            ads_test: bool, whether to return w2 expec and sin (theta) for the amplitude damped states
    '''

    # check if experimental data
    if expt and counts is not None:
        do_stokes = False
        do_counts = True
        calc_unc = False # don't explitictly calculate uncertainty in Ws for experimental data
        # assert stokes_unc is not None, "Must provide uncertainty in Stokes params"

    # if wanting to account for experimental purity, add noise to the density matrix for adjusted theoretical purity calculation
    if expt_purity is not None and angles is not None:
        rho = adjust_rho(rho, angles, expt_purity)

        # # adjust only antidiagonals
        # rho[1,2] = expt_purity*rho[1,2]
        # rho[2,1] = expt_purity*rho[2,1]
    if do_stokes or do_counts:
        if not(do_counts): expec_vals = get_expec_vals(rho)
        else:
            expec_vals = get_expec_vals_counts(counts)

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

        if calc_unc:
            def get_unc_W1(theta, stokes_unc):
                a, b = np.cos(theta), np.sin(theta)
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + ((a**2 - b**2)*stokes_unc[1,1])**2 + ((a**2 - b**2)*stokes_unc[2,2])**2 + (2*a*b)**2*(stokes_unc[3,0]**2 + stokes_unc[0,3]**2))))
            def get_unc_W2(theta, stokes_unc):
                a, b = np.cos(theta), np.sin(theta)
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + ((a**2 - b**2)*stokes_unc[1,1])**2 + ((a**2 - b**2)*stokes_unc[2,2])**2 + (2*a*b)**2*(stokes_unc[3,0]**2 + stokes_unc[0,3]**2))))
            def get_unc_W3(theta, stokes_unc):
                a, b = np.cos(theta), np.sin(theta)
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[1,1]**2 + ((a**2 - b**2)*stokes_unc[3,3])**2 + ((a**2 - b**2)*stokes_unc[2,2])**2 + (2*a*b)**2*(stokes_unc[1,0]**2 + stokes_unc[0,1]**2))))
            def get_unc_W4(theta, stokes_unc):
                a, b = np.cos(theta), np.sin(theta)
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[1,1]**2 + ((a**2 - b**2)*stokes_unc[3,3])**2 + ((a**2 - b**2)*stokes_unc[2,2])**2 + (2*a*b)**2*(stokes_unc[1,0]**2 + stokes_unc[0,1]**2))))
            def get_unc_W5(theta, stokes_unc):
                a, b = np.cos(theta), np.sin(theta)
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[2,2]**2 + ((a**2 - b**2)*stokes_unc[3,3])**2 + ((a**2 - b**2)*stokes_unc[1,1])**2 + (2*a*b)**2*(stokes_unc[2,0]**2 + stokes_unc[0,2]**2))))
            def get_unc_W6(theta, stokes_unc):
                a, b = np.cos(theta), np.sin(theta)
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[2,2]**2 + ((a**2 - b**2)*stokes_unc[3,3])**2 + ((a**2 - b**2)*stokes_unc[1,1])**2 + (2*a*b)**2*(stokes_unc[2,0]**2 + stokes_unc[0,2]**2))))
            
            ## W' from summer 2022 ##
            def get_unc_Wp1(params, stokes_unc):
                theta, alpha = params[0], params[1]
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + (np.cos(2*theta))**2*(stokes_unc[1,1]**2+stokes_unc[2,2]**2)+(np.sin(2*theta)*np.cos(alpha))**2*(stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + (np.sin(2*theta)*np.sin(alpha))**2*(stokes_unc[1,2]**2 + stokes_unc[2,1]**2))))
            def get_unc_Wp2(params, stokes_unc):
                theta, alpha = params[0], params[1]
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + (np.cos(2*theta))**2*(stokes_unc[1,1]**2+stokes_unc[2,2]**2)+(np.sin(2*theta)*np.cos(alpha))**2*(stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + (np.sin(2*theta)*np.sin(alpha))**2*(stokes_unc[1,2]**2 + stokes_unc[2,1]**2))))
            def get_unc_Wp3(params, stokes_unc):
                theta, alpha, beta = params[0], params[1], params[2]
                return np.sqrt(np.real(0.5 * (np.cos(theta)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2) + np.sin(theta)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2) + np.cos(theta)**4*np.cos(beta)**2*(stokes_unc[1,1]**2 + stokes_unc[2,2]**2) + np.sin(theta)**4*np.cos(2*alpha - beta)**2*(stokes_unc[1,1]**2 + stokes_unc[2,2]**2) + (np.sin(2*theta)*np.cos(alpha)*stokes_unc[1,0])**2 + (np.sin(2*theta)*np.cos(alpha - beta)*stokes_unc[0,1])**2 + (np.sin(2*theta)*np.sin(alpha)*stokes_unc[2,0])**2 + (np.sin(2*theta)*np.sin(alpha - beta)*stokes_unc[0,2])**2+(np.cos(theta)**2*np.sin(beta))**2*(stokes_unc[2,1]**2 + stokes_unc[1,2]**2) + np.sin(theta)**4*np.sin(2*alpha - beta)**2*(stokes_unc[2,1]**2 + stokes_unc[1,2]**2))))
            def get_unc_Wp4(params, stokes_unc):
                theta, alpha = params[0], params[1]
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2+stokes_unc[1,1]**2+np.cos(2*theta)**2*(stokes_unc[3,3]**2 + stokes_unc[2,2]**2) + (np.sin(2*theta)*np.cos(alpha))**2*(stokes_unc[0,1]**2 + stokes_unc[1,0]**2) + (np.sin(2*theta)*np.sin(alpha))**2*(stokes_unc[2,3]**2 + stokes_unc[3,2]**2))))
            def get_unc_Wp5(params, stokes_unc):
                theta, alpha = params[0], params[1]
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2+stokes_unc[1,1]**2+np.cos(2*theta)**2*(stokes_unc[3,3]**2 + stokes_unc[2,2]**2) + (np.sin(2*theta)*np.cos(alpha))**2*(stokes_unc[0,1]**2 + stokes_unc[1,0]**2) + (np.sin(2*theta)*np.sin(alpha))**2*(stokes_unc[2,3]**2 + stokes_unc[3,2]**2))))
            def get_unc_Wp6(params,stokes_unc):
                theta, alpha, beta = params[0], params[1], params[2]
                return np.sqrt(np.real(0.5*(np.cos(theta)**2*np.cos(alpha)**2*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + np.cos(theta)**4*np.sin(alpha)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + np.sin(theta)**4*np.cos(beta)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + np.sin(theta)**4*np.sin(beta)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0] **2+ stokes_unc[0,3]**2) + (np.sin(2*theta)*np.cos(alpha)*np.cos(beta))**2*(stokes_unc[1,1]**2 + stokes_unc[2,2]**2) + (np.sin(2*theta)*np.sin(alpha)*np.sin(beta))**2*(stokes_unc[1,1]**2 + stokes_unc[2,2]**2) + (np.sin(2*theta)*np.cos(alpha)*np.sin(beta))**2*(stokes_unc[2,3]**2 + stokes_unc[2,0]**2) + (np.sin(2*theta)*np.sin(alpha)*np.cos(beta))**2*(stokes_unc[2,3]**2 + stokes_unc[2,0]**2) + (np.cos(theta)**2*np.sin(2*alpha))**2*(stokes_unc[3,2]**2 + stokes_unc[0,2]**2) + (np.sin(theta)**2*np.sin(2*beta))**2*(stokes_unc[3,2]**2 + stokes_unc[0,2]**2))))
            def get_unc_Wp7(params, stokes_unc):
                theta, alpha = params[0], params[1]
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[2,2]**2+np.cos(2*theta)**2*(stokes_unc[3,3]**2 + stokes_unc[1,1]**2) + (np.sin(2*theta)*np.cos(alpha))**2*(stokes_unc[3,1]**2 + stokes_unc[1,3]**2) + (np.sin(2*theta)*np.sin(alpha))**2*(stokes_unc[2,0]**2+stokes_unc[0,2]**2))))
            def get_unc_Wp8(params, stokes_unc):
                theta, alpha = params[0], params[1]
                return np.sqrt(np.real(0.5*(stokes_unc[0,0]**2 + stokes_unc[2,2]**2 + np.cos(2*theta)**2*(stokes_unc[3,3]**2+stokes_unc[1,1]**2) + (np.sin(2*theta)*np.cos(alpha))**2*(stokes_unc[3,1]**2+stokes_unc[1,3]**2)+(np.sin(2*theta)*np.sin(alpha))**2*(stokes_unc[2,0]**2 + stokes_unc[0,2]**2))))
            def get_unc_Wp9(params, stokes_unc):
                theta, alpha, beta = params[0], params[1], params[2]
                return np.sqrt(np.real(0.5*(np.cos(theta)**4*np.cos(alpha)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + np.cos(theta)**4*np.sin(alpha)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + np.sin(theta)**4*np.cos(beta)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + np.sin(theta)**4*np.sin(beta)**4*(stokes_unc[0,0]**2 + stokes_unc[3,3]**2 + stokes_unc[3,0]**2 + stokes_unc[0,3]**2) + (np.sin(2*theta)*np.cos(alpha)*np.cos(beta))**2*(stokes_unc[1,1]**2 + stokes_unc[2,2]**2) + (np.sin(2*theta)*np.sin(alpha)*np.sin(beta))**2*(stokes_unc[1,1]**2 + stokes_unc[2,2]**2) + np.cos(theta)**4*np.sin(2*alpha)**2*(stokes_unc[0,1]**2 + stokes_unc[3,1]**2) + (np.sin(theta)**2*np.sin(2*beta))**2*(stokes_unc[0,1]**2 + stokes_unc[3,1]**2) + (np.sin(2*theta)*np.cos(alpha)*np.sin(beta))**2*(stokes_unc[1,0]**2 + stokes_unc[1,3]**2)+ (np.sin(2*theta)*np.sin(alpha)*np.cos(beta))**2*(stokes_unc[1,0]**2 + stokes_unc[1,3]**2))))


        def get_nom(params, expec_vals, func):
            '''For use in error propagation; returns the nominal value of the function'''
            w = func(params, expec_vals)
            return unp.nominal_values(w)

        # now perform optimization; break into three groups based on the number of params to optimize
        all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
        W_expec_vals = []
        if calc_unc:
            all_W_unc = [get_unc_W1, get_unc_W2, get_unc_W3, get_unc_W4, get_unc_W5, get_unc_W6, get_unc_Wp1, get_unc_Wp2, get_unc_Wp3, get_unc_Wp4, get_unc_Wp5, get_unc_Wp6, get_unc_Wp7, get_unc_Wp8, get_unc_Wp9]
        for i, W in enumerate(all_W):
            if calc_unc: get_unc_W = all_W_unc[i]
            if i <= 5: # just theta optimization
                # get initial guess at boundary
                if not(expt):
                    def min_W(x0):
                        return minimize(W, x0=x0, args=(expec_vals,), bounds=[(0, np.pi)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(expec_vals,W), bounds=[(0, np.pi/2)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x

                x0 = [0]
                w0_val = min_W_val(x0)
                w0_params = min_W_params(x0)
                x0 = [np.pi]
                w1_val = min_W_val(x0)
                w1_params = min_W_params(x0)
                if w0_val < w1_val:
                    w_min_val = w0_val
                    w_min_params = w0_params
                else:
                    w_min_val = w1_val
                    w_min_params = w1_params
                if optimize:
                    isi = 0 # index since last improvement
                    for _ in range(num_reps): # repeat 10 times and take the minimum
                        if gd:
                            if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                x0 = [np.random.rand()*np.pi]
                            else:
                                grad = approx_fprime(x0, min_W_val, 1e-6)
                                if np.all(grad < 1e-5*np.ones(len(grad))):
                                    break
                                else:
                                    x0 = x0 - zeta*grad
                        else:
                            x0 = [np.random.rand()*np.pi]

                        w_val = min_W_val(x0)
                        w_params = min_W_params(x0)
                        
                        if w_val < w_min_val:
                            w_min_val = w_val
                            w_min_params = w_params
                            isi=0
                        else:
                            isi+=1
            elif i==8 or i==11 or i==14: # theta, alpha, and beta
                if not(expt):
                    def min_W(x0):
                        return minimize(W, x0=x0, args=(expec_vals,), bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(expec_vals,W), bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x
                    
                x0 = [0, 0, 0]
                w0_val = min_W_val(x0)
                w0_params = min_W_params(x0)
                x0 = [np.pi/2, 2*np.pi, 2*np.pi]
                w1_val = min_W_val(x0)
                w1_params = min_W_params(x0)
                if w0_val < w1_val:
                    w_min_val = w0_val
                    w_min_params = w0_params
                else:
                    w_min_val = w1_val
                    w_min_params = w1_params
                if optimize:
                    isi = 0 # index since last improvement
                    for _ in range(num_reps): # repeat 10 times and take the minimum
                        if gd:
                            if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                            else:
                                grad = approx_fprime(x0, min_W_val, 1e-6)
                                if np.all(grad < 1e-5*np.ones(len(grad))):
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                else:
                                    x0 = x0 - zeta*grad
                        else:
                            x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                        w_val = min_W_val(x0)
                        w_params = min_W_params(x0)
                        # print(w_min_val, w_val)
                        if w_val < w_min_val:
                            w_min_val = w_val
                            w_min_params = w_params
                            isi=0
                        else:
                            isi+=1
                

            else:# theta and alpha
                if not(expt):
                    def min_W(x0):
                        return minimize(W, x0=x0, args=(expec_vals,), bounds=[(0, np.pi/2),(0, np.pi*2)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(expec_vals,W), bounds=[(0, np.pi/2),(0, np.pi*2)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x
                    
                x0 = [0, 0]
                w0_val = min_W(x0).fun
                w0_params = min_W(x0).x
                x0 = [np.pi/2 , 2*np.pi]
                w1_val = min_W(x0).fun
                w1_params = min_W(x0).x
                if w0_val < w1_val:
                    w_min_val = w0_val
                    w_min_params = w0_params
                else:
                    w_min_val = w1_val
                    w_min_params = w1_params
                if optimize:
                    isi = 0 # index since last improvement
                    for _ in range(num_reps): # repeat 10 times and take the minimum
                        if gd:
                            if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                            else:
                                grad = approx_fprime(x0, min_W_val, 1e-6)
                                if np.all(grad < 1e-5*np.ones(len(grad))):
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                else:
                                    x0 = x0 - zeta*grad
                        else:
                            x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                        w_val = min_W_val(x0)
                        w_params = min_W_params(x0)
                        
                        if w_val < w_min_val:
                            w_min_val = w_val
                            w_min_params = w_params
                            isi=0
                        else:
                            isi+=1

            if calc_unc:
                W_expec_vals.append(((w_min_val), get_unc_W(w_min_params, stokes_unc)))
            elif expt: # automatically calculate uncertainty
                W_expec_vals.append(W(w_min_params, expec_vals))
            else:
                W_expec_vals.append(w_min_val)
    
        # find min W expec value; this tells us if first 12 measurements are enough #
        if not(calc_unc):
            # try:
            #     W_min = np.real(min(W_expec_vals[:6]))[0] ## for some reason, on python 3.9.7 this is a list of length 1, so need to index into it. on 3.10.6 it's just a float 
            # except TypeError: # if it's a float, then just use that
            W_min = np.real(min(W_expec_vals[:6]))

            Wp_t1 = np.real(min(W_expec_vals[6:9]))
            Wp_t2 = np.real(min(W_expec_vals[9:12]))
            Wp_t3 = np.real(min(W_expec_vals[12:15]))

            return W_min, Wp_t1, Wp_t2, Wp_t3

        else: # calculate uncertainty
            # try:
            #     W_min = np.real(sorted(W_expec_vals[:6], key=lambda x:x[0])[0])[0] ## for some reason, on python 3.9.7 this is a list of length 1, so need to index into it. on 3.10.6 it's just a float 
            # except TypeError: # if it's a float, then just use that
            W_min = np.real(sorted(W_expec_vals[:6], key=lambda x:x[0])[0])

            Wp_t1 = np.real(sorted(W_expec_vals[6:9], key=lambda x: x[0])[0])
            Wp_t2 = np.real(sorted(W_expec_vals[9:12], key=lambda x: x[0])[0])
            Wp_t3 = np.real(sorted(W_expec_vals[12:15], key=lambda x: x[0])[0])

            return W_min, Wp_t1, Wp_t2, Wp_t3 # these are tuples

    else: # use operators instead like in eritas's matlab code
        # bell states #
        PHI_P = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
        PHI_M = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).reshape((4,1))
        PSI_P = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0]).reshape((4,1))
        PSI_M = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0]).reshape((4,1))
        # column vectors
        HH = np.array([1, 0, 0, 0]).reshape((4,1))
        HV = np.array([0, 1, 0, 0]).reshape((4,1))
        VH = np.array([0, 0, 1, 0]).reshape((4,1))
        VV = np.array([0, 0, 0, 1]).reshape((4,1))

        # get the operators
        # take rank 1 projector and return witness
        def get_witness(phi):
            ''' Helper function to compute the witness operator for a given state and return trace(W*rho) for a given state rho.'''
            W = phi * adjoint(phi)
            W = partial_transpose(W) # take partial transpose
            return np.real(np.trace(W @ rho))

        ## ------ for W ------ ##
        def get_W1(param):
            a,b = np.cos(param), np.sin(param)
            phi1 = a*PHI_P + b*PHI_M
            return get_witness(phi1)
        def get_W2(param):
            a,b = np.cos(param), np.sin(param)
            phi2 = a*PSI_P + b*PSI_M
            return get_witness(phi2)
        def get_W3(param):
            a,b = np.cos(param), np.sin(param)
            phi3 = a*PHI_P + b*PSI_P
            return get_witness(phi3)
        def get_W4(param):
            a,b = np.cos(param), np.sin(param)
            phi4 = a*PHI_M + b*PSI_M
            return get_witness(phi4)
        def get_W5(param):
            a,b = np.cos(param), np.sin(param)
            phi5 = a*PHI_P + 1j*b*PSI_M
            return get_witness(phi5)
        def get_W6(param):
            a,b = np.cos(param), np.sin(param)
            phi6 = a*PHI_M + 1j*b*PSI_P
            return get_witness(phi6)

        ## ------ for W' ------ ##
        def get_Wp1(params):
            theta, alpha = params[0], params[1]
            phi1_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PHI_M
            return get_witness(phi1_p)
        def get_Wp2(params):
            theta, alpha = params[0], params[1]
            phi2_p = np.cos(theta)*PSI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi2_p)
        def get_Wp3(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi3_p = 1/np.sqrt(2) * (np.cos(theta)*HH + np.exp(1j*(beta - alpha))*np.sin(theta)*HV + np.exp(1j*alpha)*np.sin(theta)*VH + np.exp(1j*beta)*np.cos(theta)*VV)
            return get_witness(phi3_p)
        def get_Wp4(params):
            theta, alpha = params[0], params[1]
            phi4_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_P
            return get_witness(phi4_p)
        def get_Wp5(params):
            theta, alpha = params[0], params[1]
            phi5_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi5_p)
        def get_Wp6(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi6_p = np.cos(theta)*np.cos(alpha)*HH + 1j * np.cos(theta)*np.sin(alpha)*HV + 1j * np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV
            return get_witness(phi6_p)
        def get_Wp7(params):
            theta, alpha = params[0], params[1]
            phi7_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi7_p)
        def get_Wp8(params):
            theta, alpha = params[0], params[1]
            phi8_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_P
            return get_witness(phi8_p)
        def get_Wp9(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi9_p = np.cos(theta)*np.cos(alpha)*HH + np.cos(theta)*np.sin(alpha)*HV + np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV
            return get_witness(phi9_p)
        
        # get the witness values by minimizing the witness function
        if not(ads_test): 
            all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
            W_expec_vals = []
            for i, W in enumerate(all_W):
                if i <= 5: # just theta optimization
                    # get initial guess at boundary
                    def min_W(x0):
                        return minimize(W, x0=x0, bounds=[(0, np.pi)])['fun']
                    x0 = [0]
                    w0 = min_W(x0)
                    x0 = [np.pi]
                    w1 = min_W(x0)
                    if w0 < w1:
                        w_min = w0
                    else:
                        w_min = w1
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        break
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                isi=0
                            else:
                                isi+=1
                elif i==8 or i==11 or i==14: # theta, alpha, and beta
                    def min_W(x0):
                        return minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])['fun']

                    x0 = [0, 0, 0]
                    w0 = min_W(x0)
                    x0 = [np.pi/2 , 2*np.pi, 2*np.pi]
                    w1 = min_W(x0)
                    if w0 < w1:
                        w_min = w0
                    else:
                        w_min = w1
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                isi=0
                            else:
                                isi+=1
                else:# theta and alpha
                    def min_W(x0):
                        return minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2)])['fun']
                        
                    x0 = [0, 0]
                    w0 = min_W(x0)
                    x0 = [np.pi/2 , 2*np.pi]
                    w1 = min_W(x0)
                    if w0 < w1:
                        w_min = w0
                    else:
                        w_min = w1
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                isi=0
                            else:
                                isi+=1

                W_expec_vals.append(w_min)
            
            # find min witness expectation values
            W_min = min(W_expec_vals[:6])
            Wp_t1 = min(W_expec_vals[6:9])
            Wp_t2 = min(W_expec_vals[9:12])
            Wp_t3 = min(W_expec_vals[12:15])

            return W_min, Wp_t1, Wp_t2, Wp_t3
        else: 
            W2_main= minimize(get_W2, x0=[0], bounds=[(0, np.pi)])
            W2_val = W2_main['fun']
            W2_param = W2_main['x']

            return W2_val, W2_param[0]


##############################################
## for entanglement verification ##
def partial_transpose(rho, subsys='B'):
    ''' Helper function to compute the partial transpose of a density matrix. Useful for the Peres-Horodecki criterion, which states that if the partial transpose of a density matrix has at least one negative eigenvalue, then the state is entangled.
    Params:
        rho: density matrix
        subsys: which subsystem to compute partial transpose wrt, i.e. 'A' or 'B'
    '''
    # decompose rho into blocks
    b1 = rho[:2, :2]
    b2 = rho[:2, 2:]
    b3 = rho[2:, :2]
    b4 = rho[2:, 2:]

    PT = np.matrix(np.block([[b1.T, b2.T], [b3.T, b4.T]]))

    if subsys=='B':
        return PT
    elif subsys=='A':
        return PT.T

def get_min_eig(rho):
    '''
    Computes the eigenvalues of the partial transpose; if at least one is negative, then state labeled as '0' for entangled; else, '1'. 
    '''

    # compute partial tranpose
    PT = partial_transpose(rho)
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
            Sy = np.array([[0,-1j],[1j,0]])
            SySy = np.kron(Sy,Sy)
            # perform spin flipping
            return SySy @ rho.conj() @ SySy
        sqrt_rho = la.sqrtm(rho)
        rho_tilde = spin_flip(rho)
        return la.sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)
    R = R_matrix(rho)
    eig_vals = np.real(la.eigvals(R))
    eig_vals = np.sort(eig_vals)[::-1] # reverse sort array
    return np.max([0,eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3]])

def check_conc_min_eig(rho, printf=False):
    ''' Returns both concurence and min eigenvalue of partial transpose. '''
    concurrence = get_concurrence(rho)
    min_eig = get_min_eig(rho)
    if printf:
        print('Concurrence: ', concurrence)
        print('Min eigenvalue: ', min_eig)
    return concurrence, min_eig

# def get_rel_entropy_concurrence(basis_key, rho):
#     ''' Based on the paper Asif et al 2023. 
#     Params:
#         basis: two character string identifer
#         rho: density matrix
#     Returns: the relative entropy of coherence'''

#     # store basis elements in dictionary
#     bases = {}
#     basis = bases[basis_key]
#     def get_rho_diag(rho):
#         rho_d = np.zeros_like(rho)
#         for s in basis:
#             rho_d += adjoint(s) @ rho @ s @ s @ adjoint(s)
#     def get_entropy(rho):
#         return -np.trace(rho @ np.log(rho))
#     rho_diag = get_rho_diag(rho)
    
#     return get_entropy(rho_diag) - get_entropy(rho)






##############################################
## for testing ##
if __name__ == '__main__':
    from random_gen import *
    import matplotlib.pyplot as plt
    def sample_reconstruct(num=100):
        ''' Samples random density matrices and reconstructs them using the 36 projections. Returns the average fidelity.'''
        fidelity_ls =[]
        for i in trange(num):
            rho = get_random_hurwitz()
            rho_prob= get_all_projs(rho)
            rho_recon = reconstruct_rho(rho_prob)
            print(is_valid_rho(rho_recon))
            fidelity_ls.append(get_fidelity(rho, rho_recon))
        plt.hist(fidelity_ls, bins=20)
        plt.show()
        print(np.mean(fidelity_ls), np.std(fidelity_ls) / np.sqrt(num))
    sample_reconstruct(100)
#     ## testing randomization processes ##
#     # random state gen imports #
#     from jones import get_random_jones
#     from random_gen import get_random_simplex, get_random_roik

#     def check_conc_min_eig_sample(N=10000, conditions=None, func=get_random_simplex, method_name='Simplex', savedir='rho_test_plots', special_name='0', display=False, fit=False):        ''' Checks random sample of N simplex generated matrices. 
#         params:
#             N: number of random states to check
#             conditions: list of conditions to check for: tuple of tuple for min max bounds for concurrence and min eigenvalue. e.g. ((0, .5), (0, -.5)) will ensure states have concurrence between 0 and 0.5 and min eigenvalue between 0 and -0.5
#             func: function to generate random state
#             method_name: name of method used to generate random state
#             savedir: directory to save plots
#             special_name: if searching with specific conditions, add a unique name to the plot
#             display: whether to display plot
#             fit: whether to fit a func to the data
#         '''
#         concurrence_ls = []
#         min_eig_ls = []
#         for n in trange(N):
#         # for n in range(N):
#             # get state
#             def get_state():
#                 if func.__name__ == 'get_random_jones' or func.__name__ == 'get_random_simplex':
#                     rho = func()[0]
#                 else:
#                     rho=func()
#                 return rho
#             # impose conditions
#             if conditions != None:
#                 go=False
#                 while not(go):
#                     rho = get_state()
#                     concurrence, min_eig = check_conc_min_eig(rho)
#                     if conditions[0][0] <= concurrence <= conditions[0][1] and conditions[1][0] <= min_eig <= conditions[1][1]:
#                         # print(is_valid_rho(state))
#                         go=True
#                     else:
#                         concurrence, min_eig = check_conc_min_eig(get_state())
#             else:
#                 # check if entangled
#                 concurrence, min_eig = check_conc_min_eig(get_state())
#             # plot
#             concurrence_ls.append(concurrence)
#             min_eig_ls.append(min_eig)

#         # fig, axes = plt.subplots(2,1, figsize=(10,5))
#         # axes[0].hist(concurrence_ls, bins=100)
#         # axes[1].hist(min_eig_ls, bins=100)
#         # plt.show()
#         plt.figure(figsize=(10,7))
#         plt.plot(concurrence_ls, min_eig_ls, 'o', label='Random states')
#         plt.xlabel('Concurrence')
#         plt.ylabel('Min eigenvalue')
#         plt.title('Concurrence vs. PT Min Eigenvalue for %s'%method_name)
        
#         if fit: # fit exponential!
#             from scipy.optimize import curve_fit
#             def func(x, a, b,c, d, e, f, g):
#                 # return a*np.exp(-b*(x+c)) + d*x**3 + e*x**2 +f*x+g
#                 return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6
#             popt, pcov = curve_fit(func, concurrence_ls, min_eig_ls)
#             perr = np.sqrt(np.diag(pcov))
#             conc_lin = np.linspace(min(concurrence_ls), max(concurrence_ls), 1000)

#             # calculate chi2red
#             # chi2red= np.sum((np.array(min_eig_ls) - func(np.array(concurrence_ls), *popt))**2)/(len(min_eig_ls) - len(pcov))
#             # plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-')
#             # plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-', label='$\lambda_{min}= %5.3f e^{-%5.3f (x+%5.3f)}+ %5.3fx^3 + %5.3fx^2 + %5.3fx + %5.3f \pm (%5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f), \chi^2_\\nu = %5.3f$'%(*popt, *perr, chi2red))
#             # plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-', label='$\lambda_{min}= %5.3f e^{-%5.3f (x+%5.3f)} %5.3fx^3  +%5.3fx^2  %5.3fx  %5.3f$'%(*popt,))
#             plt.plot(conc_lin, func(np.array(conc_lin), *popt), 'r-', label='$\lambda_{min}= %5.3f + %5.3fx + %5.3fx^2 + %5.3fx^3 + %5.3fx^4 + %5.3fx^5 + %5.3fx^6$'%(*popt,))
#             plt.legend()

#         plt.savefig('%s/conc_min_eig_%s_%s.pdf'%(savedir, method_name, special_name))

#         if display:
#             plt.show()

#     # no conditions
#     # check_conc_min_eig_sample(fit=True, method_name='Simplex', func=get_random_simplex, special_name='fit')
#     check_conc_min_eig_sample(fit=True, method_name='Roik', func=get_random_roik, special_name='fit')
#     # check_conc_min_eig_sample(fit=True, method_name='Jones', func=get_random_jones, special_name='fit')
#     # check_conc_min_eig_sample(func=get_random_roik, method_name='roik')
#     # check_conc_min_eig_sample(func=get_random_jones, method_name='jones')

#     # conditions:
#         # investigating typeI and type2 errors: type1 = concurrence = 0, min_eig < 0; type2 = concurrence > 0, min_eig > 0
#     # check_conc_min_eig_sample(N=100, method_name='jones', conditions=((0, 0), (-1000, 0)), func=get_random_jones, special_name='type1')
#     # check_conc_min_eig_sample(N=1000, method_name='roik', conditions=((0, 0), (-1000, 0)), func=get_random_roik, special_name='conc_0')
    # pass
# bigg file perform various calculations on density matrices
# methods adapted my (oscar) code as well as Alec (concurrence)

# main imports #
from os.path import join
import numpy as np
import pandas as pd
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
    try:
        fidelity = np.real((np.trace(la.sqrtm(la.sqrtm(rho1)@rho2@la.sqrtm(rho1))))**2)
        return fidelity
    except:
        print('error computing fidelity!')
        print('rho1', rho1)
        print('rho2', rho2)
        return 1e-5

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
    rho_swapped_c = rho_swapped.copy()
    rho_swapped[1, :] = rho_swapped_c[2, :]
    rho_swapped[2, :] = rho_swapped_c[1, :]

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
    h = np.array([[1], [0]])
    v = np.array([[0], [1]])
    d = np.array([[1], [1]])/np.sqrt(2)
    a = np.array([[1], [-1]])/np.sqrt(2)
    r = np.array([[1], [1j]])/np.sqrt(2)
    l = np.array([[1], [-1j]])/np.sqrt(2)

    H = np.kron(h, adjoint(h))
    V = np.kron(v, adjoint(v))
    D = np.kron(d, adjoint(d))
    A = np.kron(a, adjoint(a))
    R = np.kron(r, adjoint(r))
    L = np.kron(l, adjoint(l))

    basis_ls =[H, V, D, A, R, L]
    all_projs =[]
    for basis in basis_ls:
        for basis2 in basis_ls:
            all_projs.append(compute_roik_proj(basis, basis2, rho))

    return np.array(all_projs).reshape(6,6)

def compute_roik_proj_sc(p_1,p_2,x,m,phi):
    '''Source code from Roik et al to compute projection'''
    pp = np.kron(p_1,x)
    PP = np.kron(pp,p_2)
    result_PP= phi @ PP
    r_PP = result_PP.trace()
    cor_pp = np.kron(p_1,m)
    cor_PP = np.kron(cor_pp,p_2)
    cor_res_PP = phi @ cor_PP
    cor_r_PP = cor_res_PP.trace()
    #print(r_VV)
    #print(cor_r_VV)
    if r_PP == 0:
        fin_r_PP = 0
    else:
        fin_r_PP = r_PP/cor_r_PP
    try:
        return np.real(fin_r_PP[0][0])
    except TypeError:
        return  fin_r_PP

def get_all_roik_projs_sc(resoult):
    '''Source code from Roik et al to compute all projections'''
    # does subsystem swapping
    resoult2 = np.array([[resoult.item(0, 0),resoult.item(0,2),resoult.item(0, 1),resoult.item(0, 3)],[resoult.item(2, 0),resoult.item(2, 2),resoult.item(2,1),resoult.item(2,3)],[resoult.item(1,0),resoult.item(1,2),resoult.item(1,1),resoult.item(1,3)],[resoult.item(3,0),resoult.item(3,2),resoult.item(3,1),resoult.item(3,3)]])

    h = [[1,0],[0,0]]
    v = [[0,0],[0,1]]    
    d = [[1/2,1/2],[1/2,1/2]]
    r = [[1/2,1j/2],[-1j/2,1/2]]
    l = [[1/2,-1j/2],[1j/2,1/2]]
    a = [[1/2,-1/2],[-1/2,1/2]]
    x = [[0,0,0,0],[0,1/2,-1/2,0],[0,-1/2,1/2,0],[0,0,0,0]]
    phi = np.kron(resoult,resoult2)
    m = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] 

    projs = np.zeros((6,6))
    for i, p1 in enumerate([h, v, d, a, r, l]):
        for j, p2 in enumerate([h, v, d, a, r, l]):
            projs[i, j] = compute_roik_proj_sc(p1,p2,x,m,phi)
    return projs

def adjust_rho(rho, angles, expt_purity, state = 'E0'):
    ''' Adjusts theo density matrix to account for experimental impurity
        Multiplies unwanted experimental impurities (top right bottom right block) by expt purity to account for 
        non-entangled particles in our system '''
    if state == 'E0':
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if i < 3:
                    if j < 3:
                        pass
                if i > 2:
                    if j > 2:
                        pass
                else:
                    rho[i][j] = expt_purity * rho[i][j]
    return rho
    
    
def adjust_rho_hhvv(rho, angles, expt_purity, state = 'E0'):
    ''' Adjusts theo density matrix to account for experimental impurity
        Multiplies off-diagonal elements by expt purity to account for 
        non=entanGled particles in our system 
        
        Specifically adjusts only off diagonal elements if your entangled state only has H and V in it.
        '''
    if state =='E0':    
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if i == j:
                    pass
            else:
                rho[i][j] = expt_purity * rho[i][j]
    return rho
        
def adjust_E0_rho_general(x, rho_actual, purity, eta, chi):
    ''' Adjusts theoretical density matrix for class E0 to account for experimental impurity, but generalized to any state.
    --
    params:
        x: set of corrective probabilties
        rho-actual: target density matrix
        purity: purity of experimental state
        eta, chi: angles used to create theoretical E0 state
        model: from det_noise in process_expt.py; corresponds to number of fit params
    
    '''
    # define standard basis vecs
    HH = np.array([1,0,0,0]).reshape((4,1))
    HV = np.array([0,1,0,0]).reshape((4,1))
    VH = np.array([0,0,1,0]).reshape((4,1))
    VV = np.array([0,0,0,1]).reshape((4,1))
    HH_rho = get_rho(HH)
    HV_rho = get_rho(HV)
    VH_rho = get_rho(VH)
    VV_rho = get_rho(VV)
    # try:
    #     model = len(x)
    # except TypeError:
    #     model=1
    model = len(x)
    # implement correction    
    if model != 1:
        x /= np.sum(x)    
    if model==4:
        r_hh, r_hv, r_vh, r_vv = x
        rho_c = purity * rho_actual + (1-purity)*(r_hh*HH_rho + r_hv*HV_rho + r_vh*VH_rho + r_vv*VV_rho)

    if model==1:
        e  = x[0]
        # rho_c = (1-e1-e2)*(purity * rho_actual + (1-purity)*(((1+np.cos(chi)*np.sin(2*eta)) / 2)*HV_rho + ((1-np.cos(chi)*np.sin(2*eta)) / 2)*VH_rho)) + e1*(HH@adjoint(VV)) + e2*(VV@adjoint(HH))
        a =((1+np.cos(chi)*np.sin(2*eta)) / 2)
        b = 1-a
        # rho_c = (1-e1[0])*purity * rho_actual + (1-e1[0])*(1-purity)*(a*HV_rho + b*VH_rho) + e1[0]*(HH@adjoint(VV) + VV@adjoint(HH))

        # move HV val to HH and VH val to VV
        rho_actual_2 = rho_actual.copy()
        rho_actual_2[0,0] = rho_actual[1,1]
        rho_actual_2[1,1] = rho_actual[0,0]
        rho_actual_2[2,2] = rho_actual[3,3]
        rho_actual_2[3,3] = rho_actual[2,2]

        rho_c = (1-purity) * (1-e) *(a*HV_rho + b*VH_rho) + (1-purity) * e * (a*HH_rho + b*VV_rho) + purity * (1-e) * rho_actual + purity * e * rho_actual_2

    

    elif model==3:
        e1, e2, e3 = x
        rho_c = e3*(purity * rho_actual + (1-purity)*(((1+np.cos(chi)*np.sin(2*eta)) / 2)*HV_rho + ((1-np.cos(chi)*np.sin(2*eta)) / 2)*VH_rho)) + e1*HH_rho + e2*VV_rho
        # rho_c = e3*(purity * rho_actual + (1-purity)*(((1+np.cos(chi)*np.sin(2*eta)) / 2)*HV_rho + ((1-np.cos(chi)*np.sin(2*eta)) / 2)*VH_rho)) + e1*HH@adjoint(VV) + e2*VV @ adjoint(HH)
    # elif model==1:
    #     e=x[0]
    #     rho_c = (1-e)*(purity * rho_actual + (1-purity)*(((1+np.cos(chi)*np.sin(2*eta)) / 2)*HV_rho + ((1-np.cos(chi)*np.sin(2*eta)) / 2)*VH_rho)) + e*(purity * (get_rho((np.cos(eta)+np.exp(1j*chi)*np.sin(eta) / np.sqrt(2))*HH + (np.cos(eta)-np.exp(1j*chi)*np.sin(eta) / np.sqrt(2))*VV)) + (1-purity)*(((1+np.cos(chi)*np.sin(2*eta))/2)*HH_rho + ((1-np.cos(chi)*np.sin(2*eta))/2)*VV_rho))
    elif model==16:
        # do full correction in all bases, expanding model 4
        rho_c_1 = purity*rho_actual 
        stokes = get_expec_vals(rho_actual)
        rho_c_2 = np.dot(x, stokes.reshape((16,)))
        rho_c = rho_c_1 + (1-purity)*rho_c_2

    else:
        raise ValueError(f'model {model} value invalid')
    rho_c /= np.trace(rho_c)
    # make hermitian
    # rho_c = 0.5*(rho_c + np.conj(rho_c.T))
    # print(is_valid_rho(rho_c))
    return rho_c

def load_saved_get_E0_rho_c(rho_actual, angles_save, angles_cor, purity, model, do_W = False, do_richard = False, UV_HWP_offset = 1.029, model_path = '../../framework/decomp_test/'):
    '''Reads in data from files with optimal noise adjusements depending on extra UV_HWP offset (believed to be 1.029 degrees off during testing)
    --
    Inputs:
    rho_actual: actual rho to be corrected
    angles_save: angles used to generate the noise model
    angles_cor: angles after correcting the UVHWP
    purity: experimental purity of the state
    model: model used to generate the noise model
    do_W: 
    '''
    if model != 0 and model is not None:
        try:
            if not do_W and not do_richard:
                adjust_df = pd.read_csv(model_path+f'noise_{UV_HWP_offset}/noise_model_{model}.csv')
            elif do_W:
                adjust_df = pd.read_csv(model_path+f'noise_{UV_HWP_offset}/noise_model_{model}_W.csv')
            elif do_richard:
                adjust_df = pd.read_csv(model_path+f'noise_{UV_HWP_offset}/noise_model_{model}_richard.csv')
        except:
            print('do W', do_W)
            raise ValueError(f'You are missing the file noise_model_{model}.csv; your current path is {model_path}.')

        adjust_df = adjust_df[(np.round(adjust_df['eta'],4) == np.round(angles_save[0], 4)) & (np.round(adjust_df['chi'], 4) == np.round(angles_save[1], 4))]
        
        if model==4:
            adjust_df = adjust_df[['r_hh', 'r_hv', 'r_vh', 'r_vv']]
        elif model==3:
            adjust_df['e3'] = 1-adjust_df['e1'] - adjust_df['e2']
            adjust_df = adjust_df[['e1', 'e2', 'e3']]
        elif model==1:
            adjust_df = adjust_df[['e']]
        elif model==16:
            columns = []
            for l in list('ixyz'):
                for r in list('ixyz'):
                    columns.append(f'r_{l}{r}')
            adjust_df = adjust_df[columns]
        
        adjust_df = adjust_df.to_numpy()
        adjust_df = adjust_df.reshape((model,))
        
        adj_rho = adjust_E0_rho_general(adjust_df, rho_actual, purity, angles_cor[0], angles_cor[1])
        return adj_rho
    elif model==0 and model is not None:
        return adjust_rho(rho_actual, angles_cor, purity)
    elif model is None:
        return rho_actual

def get_adj_E0_fidelity_purity(rho, rho_actual, purity, eta, chi, model, UV_HWP_offset):
    ''' Computes the fidelity of the adjusted density matrix with the theoretical density matrix.'''
    adj_rho = load_saved_get_E0_rho_c(rho_actual, [eta, chi], purity, model, UV_HWP_offset)
    return get_fidelity(adj_rho, rho), get_purity(adj_rho)

def compute_witnesses(rho, counts = None, expt = False, verbose = True, do_counts = False, expt_purity = None, model=None, do_W = False, do_richard = False, UV_HWP_offset=None, angles = None, num_reps = 50, optimize = True, gd=True, zeta=0.7, ads_test=False, return_all=False, return_params=False, return_lynn=False, return_lynn_only=False):
    ''' Computes the minimum of the 6 Ws and the minimum of the 3 triples of the 9 W's. 
        Params:
            rho: the density matrix
            counts: raw unp array of counts and unc
            expt: bool, whether to compute the Ws assuming input is experimental data
            verbose: Whether to return which W/W' are minimal.
            do_stokes: bool, whether to compute 
            do_counts: use the raw definition in terms of counts
            expt_purity: the experimental purity of the state, which defines the noise level: 1 - purity.
            model: which model to correct for noise; see det_noise in process_expt.py for more info
            do_W: bool, whether to use W calc in loss for noise
            UV_HWP_offset: see description in det_noise in process_expt.py
            model_path: path to noise model csvs.
            angles: angles of eta, chi for E0 states to adjust theory
            num_reps: int, number of times to run the optimization
            optimize: bool, whether to optimize the Ws with random or gradient descent or to just check bounds
            gd: bool, whether to use gradient descent or brute random search
            zeta: learning rate for gradient descent
            ads_test: bool, whether to return w2 expec and sin (theta) for the amplitude damped states
            return_all: bool, whether to return all the Ws or just the min of the 6 and the min of the 3 triples
            return_params: bool, whether to return the params that give the min of the 6 and the min of the 3 triples
    '''
    # check if experimental data
    if expt and counts is not None:
        do_counts = True
    # if wanting to account for experimental purity, add noise to the density matrix for adjusted theoretical purity calculation

    # automatic correction is depricated; send the theoretical rho after whatever correction you want to this function

        # if expt_purity is not None and angles is not None: # this rho is theoretical
        #     if model is None:
        #         rho = adjust_rho(rho, angles, expt_purity)
        #     else:
        #         rho = load_saved_get_E0_rho_c(rho, angles, expt_purity, model, UV_HWP_offset, do_W = do_W, do_richard = do_richard)
        #     # rho = adjust_rho(rho, angles, expt_purity)

    if do_counts:
        counts = np.reshape(counts, (36,1))
        def get_W1(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W2(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W3(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W4(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) - 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W5(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_W6(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        
        ## W' from summer 2022 ##
        def get_Wp1(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))+((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) - ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp2(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))-((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) - np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) + ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp3(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25 * (np.cos(theta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(theta)**2*np.cos(2*alpha - beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*((DD + DA - AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.cos(alpha - beta)*((DD - DA + AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.sin(alpha)*((RR - LR + RL - LL) / (RR + LR + RL + LL)) + np.sin(2*theta)*np.sin(alpha - beta)*((RR + LR - RL - LL) / (RR + LR + RL + LL))+np.cos(theta)**2*np.sin(beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) - ((DR - DL - AR + AL) / (DR + DL + AR + AL))) + np.sin(theta)**2*np.sin(2*alpha - beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) + ((DR - DL - AR + AL) / (DR + DL + AR + AL)))))
        def get_Wp4(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1+((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((DD + DA - AD - AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp5(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1-((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((DD + DA - AD - AA) / (DD + DA + AD + AA))) - np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) + ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp6(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) + ((RR - LR + RL - LL) / (RR + LR + RL + LL))) + np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((RR - LR + RL - LL) / (RR + LR + RL + LL))) - np.cos(theta)**2*np.sin(2*alpha)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL))) - np.sin(theta)**2*np.sin(2*beta)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp7(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV))) - np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL))+((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp8(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV))-((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA))+((DH - DV - AH + AV) / (DH + DV + AH + AV)))+np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp9(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.cos(theta)**2*np.sin(2*alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(theta)**2*np.sin(2*beta)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DH - DV - AH + AV) / (DH + DV + AH + AV)))+ np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV)))))

        def get_nom(params, expec_vals, func):
            '''For use in error propagation; returns the nominal value of the function'''
            w = func(params, expec_vals)
            return unp.nominal_values(w)

        # now perform optimization; break into three groups based on the number of params to optimize
        all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
        W_expec_vals = []
        min_params = []
        for i, W in enumerate(all_W):
            if i <= 5: # just theta optimization
                # get initial guess at boundary
                if not(expt):
                    def min_W(x0):
                        return minimize(W, x0=x0, args=(counts,), bounds=[(0, np.pi)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(counts, W), bounds=[(0, np.pi)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x

                x0 = [0]
                w0_val = min_W_val(x0)
                w0_params = min_W_params(x0)
                x1 = [np.pi]
                w1_val = min_W_val(x1)
                w1_params = min_W_params(x1)
                x2 = [np.random.rand()*np.pi]
                w2_val = min_W_val(x2)
                w2_params = min_W_params(x2)
                if w0_val < w1_val and w0_val <w2_val:
                    w_min_val = w0_val
                    w_min_params = w0_params
                elif w1_val< w0_val and w1_val < w2_val:
                    w_min_val = w1_val
                    w_min_params = w1_params
                else:
                    w_min_val = w2_val
                    w_min_params = w2_params
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
                        return minimize(W, x0=x0, args=(counts,), bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(counts, W), bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])

                def min_W_val(x0):
                    return min_W(x0).fun
    
                def min_W_params(x0):
                    return min_W(x0).x
                    
                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                w0_val = min_W_val(x0)
                w0_params = min_W_params(x0)
                x1 =  [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                w1_val = min_W_val(x1)
                w1_params = min_W_params(x1)
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
                        return minimize(W, x0=x0, args=(counts,), bounds=[(0, np.pi/2),(0, np.pi*2)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(counts, W), bounds=[(0, np.pi/2),(0, np.pi*2)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x
                    
                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                w0_val = min_W(x0).fun
                w0_params = min_W(x0).x
                x1 =  [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                w1_val = min_W(x1).fun
                w1_params = min_W(x1).x
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

            if expt: # automatically calculate uncertainty
                W_expec_vals.append(W(w_min_params, counts))
            if return_params:
                min_params.append(w_min_params)
            else:
                W_expec_vals.append(w_min_val)
        W_min = np.real(min(W_expec_vals[:6]))
        try:
            Wp_t1 = np.real(min(W_expec_vals[6:9])[0])
            Wp_t2 = np.real(min(W_expec_vals[9:12])[0])
            Wp_t3 = np.real(min(W_expec_vals[12:15])[0])
        except TypeError:
            Wp_t1 = np.real(min(W_expec_vals[6:9]))
            Wp_t2 = np.real(min(W_expec_vals[9:12]))
            Wp_t3 = np.real(min(W_expec_vals[12:15]))
        
        if verbose:
            #print('i got to verbosity')
            # Define dictionary to get name of
            all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9']
            index_names = {i: name for i, name in enumerate(all_W)}
           
            W_param = [x for _,x in sorted(zip(W_expec_vals[:6], min_params[:6]))][0]
            Wp_t1_param = [x for _,x in sorted(zip(W_expec_vals[6:9], min_params[6:9]))][0]
            Wp_t2_param = [x for _,x in sorted(zip(W_expec_vals[9:12], min_params[9:12]))][0]
            Wp_t3_param = [x for _,x in sorted(zip(W_expec_vals[12:15], min_params[12:15]))][0]
           
           
            W_exp_val_ls = []
            for val in W_expec_vals:
                W_exp_val_ls.append(unp.nominal_values(val))
            
            W_min_name = [x for _,x in sorted(zip(W_exp_val_ls[:6], all_W[:6]))][0]
            Wp1_min_name = [x for _,x in sorted(zip(W_exp_val_ls[6:9], all_W[6:9]))][0]
            Wp2_min_name = [x for _,x in sorted(zip(W_exp_val_ls[9:12], all_W[9:12]))][0]
            Wp3_min_name = [x for _,x in sorted(zip(W_exp_val_ls[12:15], all_W[12:15]))][0]
            
            # print('Wp2 and its params are:', W_expec_vals[7], min_params[7])
            # print('The found W and param are:', Wp_t1, Wp1_min_name, Wp_t1_param)

            if not return_params:
                return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name
            else:
                # return same as above but with the minimum params list at end
                return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                
        else:
            return W_min, Wp_t1, Wp_t2, Wp_t3
        
        # return W_expec_vals

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
        
        def get_w_pp_a1(params):
            """
            Witness includes szx, syz, and szx
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= -a*b/c
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)
        
        def get_w_pp_a2(params):
            """
            Witness includes szx, syz, and szx
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= -a*c/b
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)

        def get_w_pp_b1(params):
            """
            Witness includes szx, syz, and szy
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= a*b/c
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)
        
        def get_w_pp_b2(params):
            """
            Witness includes szx, syz, and szy
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= a*c/b
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)
        
        def get_lynn():
            return 1/5*(2*HH +2*np.exp(1j*np.pi/4)*  HV +  np.exp(1j*np.pi/4)*VH +4*VV) 
        if return_lynn_only:
            return get_witness(get_lynn())
        # get the witness values by minimizing the witness function
        if not(ads_test): 
            all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9, get_w_pp_a1, get_w_pp_a2, get_w_pp_b1,  get_w_pp_b2]
            W_expec_vals = []
            if return_params: # to log the params
                min_params = []
            for i, W in enumerate(all_W):
                if i <= 5: # just theta optimization
                    # get initial guess at boundary
                    def min_W(x0):
                        do_min = minimize(W, x0=x0, bounds=[(0, np.pi)])
                        return do_min['fun']
                    x0 = [np.random.rand()*np.pi]
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi]
                    w1 = min_W(x1)
                    if w0 < w1:
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
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
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                    # print('------------------')
                elif i==8 or i==11 or i==14: # theta, alpha, and beta
                    def min_W(x0):
                        do_min = minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])
                        return do_min['fun']

                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                    w1 = min_W(x1)
                    if w0 < w1:
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
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
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                elif i == 15 or i==16 or i==17 or i==18: # the W'' witness
                    def min_W(x0):
                        # print(x0)
                        do_min = minimize(W, x0=x0, bounds=[(-np.pi/2 + 0.01,np.pi/2-0.01), (0.01,np.pi-0.01), (0,2*np.pi)])
                        # print(do_min['x'])
                        return do_min['fun']

                    #Begin with two random states
                    x0 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi] #generates a random set of parameters based on its relationship to beta
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi]
                    w1 = min_W(x1)
                    
                    if w0 < w1: #choose the better one 
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
                    if optimize: #Optimize the witness based on the previous best
                        isi = 0 # index since last improvement
                        count = 0
                        for _ in range(num_reps): # repeat numsteps times and take the minimum
                            count += 1
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6) #Error here assk oscar why it might be doing this>
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad          
                            else:
                                x0 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi]
                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                else:# theta and alpha
                    def min_W(x0, return_params = False):
                        if return_params == False:
                            return minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2)])['fun']
                        else:
                            return minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2)])
                        
                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                    w1 = min_W(x1)
                    if w0 < w1:
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
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
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                if return_params:
                    ### Note that these are not the correct parameters!! This must be fixed ###
                    min_params.append(x0_best)
                W_expec_vals.append(w_min)
            # print('W', np.round(W_expec_vals[:6], 3))
            # print('W\'', np.round(W_expec_vals[6:], 3))
            # find min witness expectation values
            W_min = min(W_expec_vals[:6])
            Wp_t1 = min(W_expec_vals[6:9])
            Wp_t2 = min(W_expec_vals[9:12])
            Wp_t3 = min(W_expec_vals[12:15])
            # get the corresponding parameters
            if return_params:
                W_expec_vals_ls = []
                for val in W_expec_vals:
                    W_expec_vals_ls.append(unp.nominal_values(val))
                # sort by witness value; want the most negative, so take first element in sorted
                W_param = [x for _,x in sorted(zip(W_expec_vals_ls[:6], min_params[:6]))][0]
                Wp_t1_param = [x for _,x in sorted(zip(W_expec_vals_ls[6:9], min_params[6:9]))][0]
                Wp_t2_param = [x for _,x in sorted(zip(W_expec_vals_ls[9:12], min_params[9:12]))][0]
                Wp_t3_param = [x for _,x in sorted(zip(W_expec_vals_ls[12:15], min_params[12:15]))][0]

            # calculate lynn
            W_lynn = get_witness(get_lynn())

            if not(return_all):
                if verbose:
                    #print('i got to verbosity')
                    # Define dictionary to get name of
                    all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9', 'W_pp_a1', 'W_pp_a2', 'W_pp_b1',  'W_pp_b2']
                    index_names = {i: name for i, name in enumerate(all_W)}
                
                    W_exp_val_ls = []
                    for val in W_expec_vals:
                        W_exp_val_ls.append(unp.nominal_values(val))
                    
                   
                    W_min_name = [x for _,x in sorted(zip(W_expec_vals[:6], all_W[:6]))][0]
                    Wp1_min_name = [x for _,x in sorted(zip(W_expec_vals[6:9], all_W[6:9]))][0]
                    Wp2_min_name = [x for _,x in sorted(zip(W_expec_vals[9:12], all_W[9:12]))][0]
                    Wp3_min_name = [x for _,x in sorted(zip(W_expec_vals[12:15], all_W[12:15]))][0]
                    Wpp_min_name = [x for _, x in sorted(zip(W_expec_vals[15:], all_W[15:]))][0]

                    if not return_params:
                        # Find names from dictionary and return them and their values
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name
                    else:
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                if return_params:
                    return W_min, Wp_t1, Wp_t2, Wp_t3, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                else:
                    if return_lynn:
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_lynn
                    else:
                        return W_min, Wp_t1, Wp_t2, Wp_t3
            else:
                if return_params:
                    return W_expec_vals, min_params
                else:
                    return W_expec_vals
        else: 
            print('i went to the 2nd else')
            W2_main= minimize(get_W2, x0=[0], bounds=[(0, np.pi)])
            W2_val = W2_main['fun']
            W2_param = W2_main['x']

            return W2_val, W2_param[0]

def test_witnesses():
    '''Calculate witness vals for select experimental states'''
    r1 = np.load("../../framework/decomp_test/rho_('E0', (45.0, 18.0))_32.npy", allow_pickle=True)
    counts1 = unp.uarray(r1[3], r1[4])
    print('45, 18, 32')
    print(counts1)
    print(compute_witnesses(r1[0], counts1, expt=True))
    print('------')
    r2 = np.load("../../framework/decomp_test/rho_('E0', (59.99999999999999, 72.0))_32.npy", allow_pickle=True)
    counts2 = unp.uarray(r2[3], r2[4])
    print('60, 72, 32')
    print(counts2)
    print(compute_witnesses(r2[0], counts2, expt=True))
    print('------')

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

def get_rel_entropy_concurrence(basis_key, rho):
    ''' Based on the paper Asif et al 2023. 
    Params:
        basis: two character string identifer
        rho: density matrix
    Returns: the relative entropy of coherence'''

    # store basis elements in dictionary
    bases = {}
    basis = bases[basis_key]
    def get_rho_diag(rho):
        rho_d = np.zeros_like(rho)
        for s in basis:
            rho_d += adjoint(s) @ rho @ s @ s @ adjoint(s)
    def get_entropy(rho):
        return -np.trace(rho @ np.log(rho))
    rho_diag = get_rho_diag(rho)
    
    return get_entropy(rho_diag) - get_entropy(rho)

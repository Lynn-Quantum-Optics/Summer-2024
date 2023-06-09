# file for sample jones matrix computations

# main package imports #
import numpy as np
from scipy.optimize import minimize

# special methods for density matrices #
from rho_methods import check_conc_min_eig, is_valid_rho,get_fidelity

## jones matrices ##
def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): return R(alpha) @ np.matrix(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def get_QP(phi): return np.matrix(np.diag([1, np.e**(phi*1j)]))
B = np.matrix([[0, 0, 0, 1], [1, 0,0,0]]).T
def init_state(beta, gamma): return np.matrix([[np.cos(beta),0],[0,np.e**(gamma*1j)*np.sin(beta)]])

def get_Jrho_C(angles):
    ''' Jones matrix with *almost* current setup, just adding one QWP on Alice. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_Detectors
        params:
            angles: list of angles for HWP1, HWP2, QWP1, QWP2, QP
            beta, gamma: angles for init polarization state
    '''
    Is = init_state(angles[0], angles[1])
    H1 = H(angles[2])
    H2 = H(angles[3])
    Q1 = Q(angles[4])
    Q2 = Q(angles[5])
    QP = get_QP(angles[6])

    ## compute density matrix ##
    P = np.kron(Q2, Q1 @ H2) @ B @ QP @ H1 @ Is
    rho = np.round(P @ P.H,2)

    return rho

def get_Jrho_I(angles):
    ''' Jones matrix method with Ideal setup. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_HWP -> A_QWP -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_QWP -> B_Detectors

        params:
            angles: list of angles for HWP1, HWP2, HWP3, QWP1, QWP2, QWP3, QWP4, QWP5, QWP6
            beta, gamma: angles for init polarization state
    '''
    Is = init_state(angles[0], angles[1])
    H1 = H(angles[2])
    H2 = H(angles[3])
    H3 = H(angles[4])
    Q1 = Q(angles[5])
    Q2 = Q(angles[6])
    Q3 = Q(angles[7])
    Q4 = Q(angles[8])
    Q5 = Q(angles[9])
    Q6 = Q(angles[10])
    

    ## compute density matrix ##
    P = np.kron(H3 @ Q5 @ Q6, H2 @ Q3 @ Q4) @ B @Q2 @Q1 @ H1 @ Is
    rho = np.round(P @ P.H,2)

    return rho

def get_random_Jangles(setup='C'):
    ''' Returns random angles for the Jrho_C setup
    params: setup: either 'C' for current setup or 'I' for ideal setup
    '''
    # init state
    beta = np.random.rand()*np.pi/2 
    gamma = np.random.rand()*2*np.pi

    if setup == 'C':
        # HWPs
        theta_ls = np.random.rand(2)*np.pi/4
        # QWPs
        alpha_ls = np.random.rand(2)*np.pi/2
        #QP
        phi = np.random.rand()*0.69 # experimental limit of our QP
        return [beta, gamma, *theta_ls, *alpha_ls, phi]

    if setup =='I':
        # HWPs
        theta_ls = np.random.rand(3)*np.pi/4
        # QWPs
        alpha_ls = np.random.rand(6)*np.pi/2
        return [beta, gamma, *theta_ls, *alpha_ls]

def get_random_jones(setup='C'):
    ''' Computes random angles in the ranges specified and generates the resulting states'''
    # get random angles
    angles = get_random_Jangles(setup=setup)
    # get density matrix
    if setup=='C':
        rho = get_Jrho_C(angles)
    elif setup=='I':
        rho = get_Jrho_I(angles)
    else:
        raise ValueError('Invalid setup. Must be either "C" or "I"')

    # call method to confirm state is valid
    while not(is_valid_rho(rho)):
        angles = get_random_Jangles()
        rho = get_Jrho_C(angles)

    return [rho, angles]

def jones_decompose(targ_rho, setup = 'C', eps_min=0.8, eps_max=0.95, N = 20000, verbose=False):
    ''' Function to decompose a given density matrix into jones matrices
    params:
        targ_rho: target density matrix
        setup: either 'C' for current setup or 'I' for ideal setup
        eps_min: minimum tolerance for fidelity
        eps_max: maximum tolerance for fidelity; if reached, halleljuah and break early!
        N: max number of times to try to optimize
        verbose: whether to include print statements.
    returns:
        angles: list of angles matching setup return. note beta and gamma, which set the initial polarization state, are the first two elements
        fidelity: fidelity of the resulting guessed state
    note: if you use Cntrl-C to break out of the function, it will return the best guess so far
    '''
    
    # initial guesses (PhiP)
    if setup=='C':
        func = get_Jrho_C
        # x0 = [np.pi/8,0,0, 0, 0]
        bounds = [(0, np.pi/2), (0, 2*np.pi), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2), (0, 0.69)]
    elif setup=='I':
        func = get_Jrho_I
        bounds = [(0, np.pi/2), (0, 2*np.pi), (0, np.pi/4), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2),(0, np.pi/2), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2) ]
    else:
        raise ValueError('Invalid setup. Must be either "C" or "I"')

    def loss_fidelity(angles, targ_rho):
        ''' Function to quantify the distance between the targ and pred density matrices'''
        pred_rho = func(angles)
        fidelity = get_fidelity(pred_rho, targ_rho)
        
        return 1-np.sqrt(fidelity)

    # get initial result
    x0 = get_random_Jangles(setup=setup)
    result = minimize(loss_fidelity, x0=x0, args=(targ_rho,), bounds=bounds)
    best_angles = result.x
    fidelity = get_fidelity(func(best_angles), targ_rho)
    rho = func(best_angles)
    
    # iterate eiher until we hit max bound or we get a valid rho with fidelity above the min
    max_best_fidelity = fidelity
    max_best_angles = best_angles
    n=0
    while n < N and (not(is_valid_rho(rho)) or  (is_valid_rho(rho) and fidelity<eps_min)):
        try:
            n+=1
            if verbose: 
                print('n', n)
                print(fidelity, max_best_fidelity)
            
            # start with different initial guesses
            x0 = get_random_Jangles(setup=setup)
            result = minimize(loss_fidelity, x0=x0, args=(targ_rho,), bounds=bounds, tol=1e-10, method='TNC')
            best_angles = result.x
            fidelity = get_fidelity(func(best_angles), targ_rho)
            rho = func(best_angles)

            if fidelity > max_best_fidelity:
                max_best_fidelity = fidelity
                max_best_angles = best_angles

            if fidelity > eps_max: # if we get a good enough result, break early
                break

        except KeyboardInterrupt:
            print('interrupted...')
            break
            
    if verbose:
        print('actual state', targ_rho)
        print('predicted state', func(best_angles) )
        print('fidelity', fidelity)
    return max_best_angles, max_best_fidelity


if __name__=='__main__':
    import pandas as pd
    from tqdm import trange

    # import predefined states for testing
    from sample_rho import *
    from random_gen import *

    # initilize dataframe to store results
    decomp_df = pd.DataFrame(columns=['state', 'angles', 'fidelity'])

    # define test states
    # get random eta, chi: 
    eta_ls = np.random.rand(3)*np.pi/2
    chi_ls = np.random.rand(3)*2*np.pi
    states_C = [PhiP, PhiM, PsiP, PsiM, E_state0(eta_ls[0], chi_ls[0]), E_state0(eta_ls[1], chi_ls[1]), E_state0(eta_ls[2], chi_ls[2])]
    states_names_C = ['PhiP', 'PhiM', 'PsiP', 'PsiM', 'E0_'+str(eta_ls[0])+'_'+str(chi_ls[0]), 'E0_'+str(eta_ls[1])+'_'+str(chi_ls[1]), 'E0_'+str(eta_ls[2])+'_'+str(chi_ls[2])]
    setup_C = ['C', 'C', 'C', 'C', 'C', 'C', 'C']
    states_I = [E_state0(eta_ls[0], chi_ls[0]), E_state0(eta_ls[1], chi_ls[1]), E_state0(eta_ls[2], chi_ls[2])]
    setup_I = ['I', 'I', 'I']
    states_names_I= ['E0_'+str(eta_ls[0])+'_'+str(chi_ls[0]), 'E0_'+str(eta_ls[1])+'_'+str(chi_ls[1]), 'E0_'+str(eta_ls[2])+'_'+str(chi_ls[2])]

    states_tot = states_C + states_I
    names_tot = states_names_C + states_names_I
    setup_tot = setup_C + setup_I

    # run decomposition
    for i in trange(len(states_tot)):
        print('starting state', names_tot[i], '...')
        angles, fidelity = jones_decompose(states_tot[i], setup_tot[i], verbose=True)
        decomp_df = decomp_df.append({'state': names_tot[i], 'angles': angles, 'fidelity': fidelity}, ignore_index=True)
        
    # save results
    decomp_df.to_csv('decomp_results.csv')
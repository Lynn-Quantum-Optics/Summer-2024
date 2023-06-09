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

def jones_decompose(setup, targ_rho, eps):
    ''' Function to decompose a given density matrix into jones matrices. 
    params:
        setup: either 'C' for current setup or 'I' for ideal setup
        targ_rho: target density matrix
        eps: tolerance for fidelity
    '''
    
    # initial guesses (PhiP)
    if setup=='C':
        func = get_Jrho_C
        # x0 = [np.pi/8,0,0, 0, 0]
        x0 = get_random_Jangles(setup=setup)
        bounds = [(0, np.pi/2), (0, 2*np.pi), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2), (0, 0.69)]
    elif setup=='I':
        func = get_Jrho_I
        x0 = get_random_Jangles(setup=setup)
        bounds = [(0, np.pi/2), (0, 2*np.pi), (0, np.pi/4), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2),(0, np.pi/2), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2) ]
    else:
        raise ValueError('Invalid setup. Must be either "C" or "I"')

    def loss_fidelity(angles, targ_rho):
        pred_rho = func(angles)
        fidelity = get_fidelity(pred_rho, targ_rho)

        return np.sqrt(2*(1-np.sqrt(fidelity)))

    # minimize loss function
    # min_result= minimize(loss_frobenius, x0=x0, args=(targ_rho), bounds=bounds, tol=1e-10, method='TNC')
    result = minimize(loss_fidelity, x0=x0, args=(targ_rho,), bounds=bounds)
    min_loss = result.fun
    best_angles = result.x
    fidelity = get_fidelity(func(best_angles), targ_rho)
    N=10**9 # max number of times to try
    n=0
    while n < N and fidelity<eps:
        try:
            while not(is_valid_rho(func(best_angles))) and fidelity<eps:
                print('invalid rho, trying again')
                x0 = get_random_Jangles(setup=setup)
                result = minimize(loss_fidelity, x0=x0, args=(targ_rho,), bounds=bounds, tol=1e-10, method='TNC')
                min_loss = result.fun
                best_angles = result.x
                fidelity = get_fidelity(func(best_angles), targ_rho)
        except ValueError: # in case the matrix is all 0, run again
            print('invalid rho, trying again')
            x0 = get_random_Jangles(setup=setup)
            result = minimize(loss_fidelity, x0=x0, args=(targ_rho,), bounds=bounds, tol=1e-10, method='TNC')
            min_loss = result.fun
            best_angles = result.x
            fidelity = get_fidelity(func(best_angles), targ_rho)
        n+=1


    print('actual state', targ_rho)
    print('predicted state', func(best_angles) )
    # print('angles', best_angles)
    print('loss', min_loss)
    print('fidelity', get_fidelity(func(best_angles), targ_rho))
    return best_angles, min_loss, fidelity


if __name__=='__main__':
    # import predefined states for testing
    from sample_rho import PhiP, PhiM, PsiP, PsiM, PhiPM
    from random_gen import get_random_simplex
    from scipy.stats import sem

    # angles=[np.pi/8,0,0, 0, 0] # PhiP
    # angles=[np.pi/8, np.pi/4, 0, 0, np.pi] # PsiM
    # rho = get_Jrho_C(angles)
    # print(rho)
    fidelity_ls =[]
    for l in range(10):
        best_angles, min_loss, fidelity= jones_decompose('C', get_random_jones()[0], 0.85)
        fidelity_ls.append(fidelity)
    print(np.mean(fidelity_ls), sem(fidelity_ls))
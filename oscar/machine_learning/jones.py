# file for sample jones matrix computations

# main package imports #
import numpy as np
from scipy.optimize import minimize

# special methods for density matrices #
from rho_methods import *
from random_gen import *

## jones matrices ##
def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): return R(alpha) @ np.matrix(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def get_QP(phi): return np.matrix(np.diag([1, np.e**(phi*1j)]))
B = np.matrix([[0, 0, 0, 1], [1, 0,0,0]]).T
def init_state(beta, gamma): return np.matrix([[np.cos(beta),0],[0,np.e**(gamma*1j)*np.sin(beta)]])

def get_Jrho_C(angles, check=False):
    ''' Jones matrix with *almost* current setup, just adding one QWP on Alice. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_Detectors
        params:
            angles: list of angles for init polarization state (beta, gamma), HWP1, HWP2, QWP1, QWP2, QP
            check: boolean to check if density matrix is valid; don't call when minimizing bc throwing an error will distrupt minimzation process. the check is handled in the while statement in jones_decomp -- for this reason, set to False by default
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

    if check:
        if is_valid_rho(rho):
            return rho
        else:
            print(rho)
            raise ValueError('Invalid density matrix')

    else:
        return rho

def get_Jrho_I(angles, check=False):
    ''' Jones matrix method with Ideal setup. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_HWP -> A_QWP -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_QWP -> B_Detectors

        params:
            angles: list of angles for init polarzation state (beta, gamma), HWP1, HWP2, HWP3, QWP1, QWP2, QWP3, QWP4, QWP5, QWP6
            check: boolean to check if density matrix is valid; don't call when minimizing bc throwing an error will distrupt minimzation process. the check is handled in the while statement in jones_decomp -- for this reason, set to False by default
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

    if check:
        if is_valid_rho(rho):
            return rho
        else:
            print(rho)
            raise ValueError('Invalid density matrix')

    else:
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

def jones_decompose(targ_rho, setup = 'C', eps_max=0.99, N = 30000, verbose=False):
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
        raise ValueError('Invalid setup. Must be either "C" or "I". You have f{setup}.')

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
    if is_valid_rho(rho):
        max_best_fidelity = fidelity
        max_best_angles = best_angles
    else:
        max_best_fidelity = 0
        max_best_angles = []
    n=0
    while n < N and (not(is_valid_rho(rho)) or  (is_valid_rho(rho) and fidelity<eps_max)):
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

            if (fidelity > max_best_fidelity) and is_valid_rho(rho): # have to be a valid density matrix to assign maxes
                max_best_fidelity = fidelity
                max_best_angles = best_angles

        except KeyboardInterrupt:
            print('interrupted...saving best result so far')
            break
            
    if verbose:
        print('actual state', targ_rho)
        print('predicted state', func(max_best_angles) )
        print('fidelity', max_best_fidelity)
        print('projections', get_all_projections(func(max_best_angles)))

    # compute projections in 12 basis states
    HH, HV, VH, VV, DD, DA, AD, AA, RR, RL, LR, LL = get_all_projections(func(max_best_angles))

    return max_best_angles, max_best_fidelity, [HH, HV, VH, VV, DD, DA, AD, AA, RR, RL, LR, LL], n


if __name__=='__main__':
    import pandas as pd
    from tqdm import trange

    # import predefined states for testing
    from sample_rho import *
    from random_gen import *

    def do_full_ex_decomp(setup,savename,bell=True, eritas=True, random=True, num_random=10, N=30000):
        ''' Run example decompositions using the C setup.
        Params:
            setup: 'C' or ;'I'
            savename: name of file to save results to
            bell: whether to include bell states
            num_random: number of random states to decompose
            (optinonal) num_random: number of random states to decompose
        '''
        # initilize dataframe to store results
        decomp_df = pd.DataFrame({'state':[], 'n':[], 'setup': [], 'init_pol':[], 'angles':[], 'fidelity':[], 'projH&V':[], 'projD&A':[], 'projR&L':[]})
        states = []
        states_names = []

        if bell:
            states_bell = [PhiP, PhiM, PsiP, PsiM]
            states_bell_names = ['PhiP', 'PhiM', 'PsiP', 'PsiM']
            states += states_bell
            states_names += states_bell_names

        if eritas:
            # get random Eritas states
            states_E0_all = [get_random_E0() for i in range(num_random)]
            states_E0 = [states_E0_all[i][0] for i in range(num_random)]
            states_E0_names = [f'E0_{states_E0_all[i][1][0]}_{states_E0_all[i][1][1]}' for i in range(num_random)]

            states+=states_E0
            states_names+=states_E0_names

            states_E1_all = [get_random_E1() for i in range(num_random)]
            states_E1 = [states_E1_all[i][0] for i in range(num_random)]   
            states_E1_names = [f'E1_{states_E1_all[i][1][0]}_{states_E1_all[i][1][1]}' for i in range(num_random)]

            states+=states_E1
            states_names+=states_E1_names
         
        if random:
            states_RS_all = [get_random_simplex() for i in range(num_random)]
            states_RS = [states_RS_all[i][0] for i in range(num_random)]
            states_RS_names = [f'RS_{states_RS_all[i][1][0]}_{states_RS_all[i][1][1]}_{states_RS_all[i][1][2]}_{states_RS_all[i][1][3]}__{states_RS_all[i][1][4]}__{states_RS_all[i][1][5]}__{states_RS_all[i][1][6]}' for i in range(num_random)]
       
            states+=states_RS
            states_names+=states_RS_names

        for i in trange(len(states)):
            try:
                angles, fidelity, projections, n = jones_decompose(states[i], setup, N=N, verbose=True)
                decomp_df = pd.concat([decomp_df, pd.DataFrame.from_records([{'state':states_names[i], 'n':n, 'setup': setup, 'init_pol':angles[:2],'angles':angles[2:], 'fidelity':fidelity, 'projH&V':projections[:4], 'projD&A':projections[4:8], 'projR&L':projections[8:]}])])
            except: # if there's an error, save what we have
                print('error! saving what we have')
                decomp_df.to_csv(savename+'.csv')
                
        decomp_df.to_csv(savename+'.csv')
        return decomp_df
    setup=input('Enter setup (C or I): ')
    while setup != 'C' and setup != 'I':
        setup=input('Enter setup (C or I): ')
    bell = bool(int(input('include bell states?')))
    eritas = bool(int(input('include eritas states?')))
    random = bool(int(input('include random states?')))
    special = input('sepical name to append to file?')
    savename = f'decomp_{setup}_{bell}_{eritas}_{random}_{special}'
    print(setup, bell, eritas, random)

    do_full_ex_decomp('I',bell=bell, eritas=eritas, random=random, savename=savename)
        
     # define test states
    # get random eta, chi: 
    # num_random = 10
    # eta_ls = np.random.rand(num_random)*np.pi/2
    # chi_ls = np.random.rand(num_random)*2*np.pi

    # states=[PhiP, PhiM, PsiP, PsiM, *[E_state0(eta_ls[i], chi_ls[i]) for i in range(num_random)], *[E_state1(eta_ls[i], chi_ls[i]) for i in range(num_random)], *[get_random_simplex()[0] for i in range(num_random)]]
    # states_names = ['PhiP', 'PhiM', 'PsiP', 'PsiM', *[f'E0_{eta_ls[i]}_{chi_ls[i]}' for i in range(num_random)], *[f'E1_{eta_ls[i]}_{chi_ls[i]}' for i in range(num_random)], *[f'RS_{get_random_simplex()[1]}' for i in range(num_random)]]
    # # do only C setup for now
    # setup = ['C', 'C', 'C', 'C', *['C' for i in range(num_random)], *['C' for i in range(num_random)], *['C' for i in range(num_random)]]

     # run decomposition
    # for i in trange(len(states)):
    #     print('starting state', states_names[i], '...')
    #     angles, fidelity = jones_decompose(states[i], setup[i], verbose=True)
    #     decomp_df = pd.concat([decomp_df, pd.DataFrame.from_records([{'state': states_names[i], 'angles': angles, 'fidelity': fidelity}])])

    # states_C = [PhiP, PhiM, PsiP, PsiM, E_state0(eta_ls[0], chi_ls[0]), E_state0(eta_ls[1], chi_ls[1]), E_state0(eta_ls[2], chi_ls[2])]
    # states_names_C = ['PhiP', 'PhiM', 'PsiP', 'PsiM', 'E0_'+str(eta_ls[0])+'_'+str(chi_ls[0]), 'E0_'+str(eta_ls[1])+'_'+str(chi_ls[1]), 'E0_'+str(eta_ls[2])+'_'+str(chi_ls[2])]
    # setup_C = ['C', 'C', 'C', 'C', 'C', 'C', 'C']
    # states_I = [E_state0(eta_ls[0], chi_ls[0]), E_state0(eta_ls[1], chi_ls[1]), E_state0(eta_ls[2], chi_ls[2])]
    # setup_I = ['I', 'I', 'I']
    # states_names_I= ['E0_'+str(eta_ls[0])+'_'+str(chi_ls[0]), 'E0_'+str(eta_ls[1])+'_'+str(chi_ls[1]), 'E0_'+str(eta_ls[2])+'_'+str(chi_ls[2])]

    # states_tot = states_C + states_I
    # names_tot = states_names_C + states_names_I
    # setup_tot = setup_C + setup_I

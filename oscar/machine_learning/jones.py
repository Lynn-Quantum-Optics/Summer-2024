# file for jones matrix computations

# main package imports #
import numpy as np
from scipy.optimize import minimize

# special methods for density matrices #
from rho_methods import *
from random_gen import *

#### jones matrices ####
def R(alpha): 
    ''' Rotation matrix for angle alpha'''
    return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): 
    ''' HWP matrix for angle theta'''
    return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): 
    ''' QWP matrix for angle alpha'''
    return R(alpha) @ np.matrix(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def get_QP(phi): 
    ''' QP matrix for angle phi'''
    return np.matrix(np.diag([1, np.e**(phi*1j)]))
### BBO matrix ###
BBO = np.matrix([[0, 0, 0, 1], [1, 0,0,0]]).T
def init_state_bg(beta, gamma): 
    ''' Initial polarization state matrix for angles beta, gamma: |psi> = cos(beta)|H> + e^(gamma*i)sin(beta)|V>'''
    return np.matrix([[np.cos(beta),0],[0,np.e**(gamma*1j)*np.sin(beta)]])
# initial state for simple=False
Is0 = init_state_bg(0,0)

def get_Jrho(angles, setup = 'C', simple=True, check=False):
    ''' Main function to get density matrix using Jones Matrix setup.
    Params:
        angles: list of angles for setup. see the conditionals for specifics
        setup: either 'C' for current setup or 'I' for ideal setup
        simple: boolean for whether to start with arbitrary cos(beta)|0> + sin(beta) e^(i*gamma)|1> or to use a combination of HWP and QP or HWP and 2x QP for C and I respectively
        check: boolean to check if density matrix is valid; don't call when minimizing bc throwing an error will distrupt minimzation process. the check is handled in the while statement in jones_decomp -- for this reason, set to False by default
    '''

    if setup == 'C':
        ''' Jones matrix with *almost* current setup, just adding one QWP on Alice. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_Detectors
        '''

        if simple:
            ''' Assumes input state of cos(beta)|0> + sin(beta) e^(i*gamma)|1>. Input angles:
                beta, gamma, Bob's HWP, Bob's QWP, Alice's QWP'''

            # init state
            beta = angles[0]
            gamma = angles[1]

            # B HWP
            B_theta = angles[2]
            # B and A QWPs
            B_alpha = angles[3]
            A_alpha = angles[4]

            Is = init_state_bg(beta, gamma)
            H_B = H(B_theta)
            Q_B = Q(B_alpha)
            Q_A = Q(A_alpha)

            P = np.kron(Q_A, Q_B @ H_B) @ BBO @ Is
            rho = P @ P.H

        else:
            ''' Starts with input state with beta = gamma = 0. Input angles:
                UV_HWP, QP, Bob's HWP, Bob's QWP, Alice's QWP
            '''
            # UVHWP 
            theta0 = angles[0]
            # QP
            phi = angles[1]
            # B HWP
            B_theta = angles[2]
            # B and A QWPs
            B_alpha = angles[3]
            A_alpha = angles[4]

            H_UV = H(theta0)
            QP = get_QP(phi)
            H_B = H(B_theta)
            Q_B = Q(B_alpha)
            Q_A = Q(A_alpha)

            P = np.kron(Q_A, Q_B @ H_B) @ BBO @ QP @ H_UV @ Is0
            rho = P @ P.H

    elif setup == 'I':
        ''' Jones matrix method with Ideal setup. Computes the rotated polarization state using the following setup:
            UV_HWP -> QP -> BBO -> A_HWP -> A_QWP -> A_QWP -> A_Detectors
                                -> B_HWP -> B_QWP -> B_QWP -> B_Detectors
        '''
        
        if simple:
            ''' Assumes input state of cos(beta)|0> + sin(beta) e^(i*gamma)|1>. Input angles:
                beta, gamma, Bob's HWP, Alice's HWP, Bob's QWP1, Bob's QWP2, Alice's QWP1, Alice's QWP2'''
            # init state
            beta = angles[0]
            gamma = angles[1]
            # HWPs
            B_theta = angles[2]
            A_theta = angles[3]
            # B and A QWPs
            B_alpha_ls = angles[3:5]
            A_alpha_ls = angles[5:7]

            Is = init_state_bg(beta, gamma)
            H_B = H(B_theta)
            H_A = H(A_theta)
            Q_B1 = Q(B_alpha_ls[0])
            Q_B2 = Q(B_alpha_ls[1])
            Q_A1 = Q(A_alpha_ls[0])
            Q_A2 = Q(A_alpha_ls[1])

            P = np.kron(Q_A1 @ Q_A2 @ H_A, Q_B1 @ Q_B2 @ H_B) @ BBO @ Is
            rho = P @ P.H

        else:
            ''' Starts with input state with beta = gamma = 0. Input angles:
            UV_HWP, QP0, QP1, Bob's HWP, Alice's HWP, Bob's QWP1, Bob's QWP2, Alice's QWP1, Alice's QWP2
            '''
            # UVHWP 
            theta_0 = angles[0]
            # initial QWPs
            Q_init = angles[1:3]
            # HWPs
            theta_B = angles[3]
            theta_A = angles[4]
            # B and A QWPs
            alpha_B_ls = angles[5:7]
            alpha_A_ls = angles[7:9]

            H_UV = H(theta_0)
            Q0= Q(Q_init[0])
            Q1 = Q(Q_init[1])
            H_B = H(theta_B)
            H_A = H(theta_A)
            Q_B1 = Q(alpha_B_ls[0])
            Q_B2 = Q(alpha_B_ls[1])
            Q_A1 = Q(alpha_A_ls[0])
            Q_A2 = Q(alpha_A_ls[1])

            P = np.kron(Q_A1 @ Q_A2 @ H_A, Q_B1 @ Q_B2 @ H_B) @ BBO @ Q1 @ Q0 @ H_UV @ Is0

            rho = P @ P.H

    else:
        raise ValueError(f'Invalid setup. You have {setup} but needs to be either "C" or "I".')

    ## return density matrix ##
    if check:
        if is_valid_rho(rho):
            return rho
        else:
            print(rho)
            raise ValueError('Invalid density matrix')

    else:
        return rho



def get_random_Jangles(setup='C', simple=True):
    ''' Returns random angles for the Jrho_C setup. Confirms that the density matrix is valid.
    params: setup: either 'C' for current setup or 'I' for ideal setup
            simple: boolean for whether to start with arbitrary cos(beta)|0> + sin(beta) e^(i*gamma)|1> or to use a combination of HWP and QP or HWP and 2x QP
    '''

    def get_angles():

        if setup=='C':
            if simple:
                # init state
                beta = np.random.rand()*np.pi/2 
                gamma = np.random.rand()*2*np.pi

                # HWP
                theta = np.random.rand()*np.pi/4
                # QWPs
                alpha_ls = np.random.rand(2)*np.pi/2
                angles = [beta, gamma, theta, *alpha_ls]

            else:
                # UV HWP
                theta_UV = np.random.rand()*np.pi/4
                # QP
                phi = np.random.rand()*0.69
                # B HWP
                theta_B = np.random.rand()*np.pi/4
                # QWPs
                alpha_ls = np.random.rand(2)*np.pi/2
                angles= [theta_UV, phi, theta_B, *alpha_ls]
            
        elif setup =='I':
            if simple:
                # init state
                beta = np.random.rand()*np.pi/2 
                gamma = np.random.rand()*2*np.pi
                # HWPs
                theta_ls = np.random.rand(2)*np.pi/4
                # QWPs
                alpha_ls = np.random.rand(4)*np.pi/2
                angles = [beta, gamma, *theta_ls, *alpha_ls]
            else:
                # UVHWP 
                theta_0 = np.random.rand()*np.pi/4
                # initial QWPs
                Q_init = np.random.rand(2)*np.pi/2
                # HWPs
                theta_ls = np.random.rand(2)*np.pi/4
                # B and A QWPs
                alpha_ls = np.random.rand(4)*np.pi/2

                angles= [theta_0, *Q_init, *theta_ls, *alpha_ls]

        return angles

    # confirm angles are valid #
    angles = get_angles()
    while not(is_valid_rho(get_Jrho(angles=angles, setup=setup, simple=simple))):
        angles = get_angles()
    return angles


def get_random_jones(setup='C', simple=True):
    ''' Computes random angles in the ranges specified and generates the resulting states'''
    # get random angles
    angles = get_random_Jangles(setup=setup)
    # get density matrix
    rho = get_Jrho(setup=setup, angles=angles, simple=simple)

    return [rho, angles]

def jones_decompose(targ_rho, setup = 'C', simple=True, eps_max=0.99, N = 1000, verbose=False):
    ''' Function to decompose a given density matrix into jones matrices
    params:
        targ_rho: target density matrix
        setup: either 'C' for current setup or 'I' for ideal setup
        simple: boolean for whether to start with arbitrary cos(beta)|0> + sin(beta) e^(i*gamma)|1> or to use a combination of HWP and QP or HWP and 2x QP
        eps_min: minimum tolerance for fidelity
        eps_max: maximum tolerance for fidelity; if reached, halleljuah and break early!
        N: max number of times to try to optimize
        verbose: whether to include print statements.
    returns:
        angles: list of angles matching setup return. note beta and gamma, which set the initial polarization state, are the first two elements
        fidelity: fidelity of the resulting guessed state
    note: if you use Cntrl-C to break out of the function, it will return the best guess so far
    '''
    
    func = lambda angles: get_Jrho(angles=angles, setup=setup, simple=simple)

    # initial guesses (PhiP)
    if setup=='C':
        if simple: # beta, gamma, theta, alpha1, alpha2
            bounds = [(0, np.pi/2), (0, 2*np.pi), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2)]
        else: # theta_0, phi, theta_B, alpha_B1, alpha_B2
            bounds = [(0, np.pi/4), (0, 0.69), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2)]
    elif setup=='I':
        if simple: # beta, gamma, theta1, theta2, alpha1, alpha2
            bounds = [(0, np.pi/2), (0, 2*np.pi), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2)]
        else: # theta_uv, alpha_0, alpha_1, theta_B, theta_A, alpha_B1, alpha_B2, alpha_A1, alpha_A2
            bounds = [(0, np.pi/4),  (0, np.pi/2), (0, np.pi/2), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2)]
    else:
        raise ValueError(f'Invalid setup. Must be either "C" or "I". You have {setup}.')

    def loss_fidelity(angles, targ_rho):
        ''' Function to quantify the distance between the targ and pred density matrices'''
        pred_rho = func(angles)
        fidelity = get_fidelity(pred_rho, targ_rho)
        
        return 1-np.sqrt(fidelity)

    # get initial result
    def guess_angles():
        x0 = get_random_Jangles(setup=setup, simple=simple)
        # print(len(x0), x0)
        # print(len(bounds), bounds)
        result = minimize(loss_fidelity, x0=x0, args=(targ_rho,), bounds=bounds)
        best_angles = result.x
        rho = func(best_angles)
        fidelity = get_fidelity(rho, targ_rho)
        return best_angles, fidelity, rho

    best_angles, fidelity, rho = guess_angles()
    
    # iterate eiher until we hit max bound or we get a valid rho with fidelity above the min
    if is_valid_rho(rho):
        max_best_fidelity = fidelity
        max_best_angles = best_angles
    else:
        max_best_fidelity = 0
        max_best_angles = []
    n=0
    while n < N and fidelity<eps_max:
        try:
            n+=1
            if verbose: 
                print('n', n)
                print(fidelity, max_best_fidelity)
            
            # start with different initial guesses
            best_angles, fidelity, rho = guess_angles()

            if fidelity > max_best_fidelity:
                max_best_fidelity = fidelity
                max_best_angles = best_angles

        except KeyboardInterrupt:
            print('interrupted...saving best result so far')
            break

    # compute projections in 12 basis states
    proj_pred = [get_all_projections(func(max_best_angles))]
    proj_targ = [get_all_projections(targ_rho)]

    if verbose:
        print('actual state', targ_rho)
        print('predicted state', func(max_best_angles) )
        print('num iterations', n)
        print('fidelity', max_best_fidelity)
        print('projections of predicted', proj_pred)
        print('projections of actual', proj_targ)

    return max_best_angles, max_best_fidelity, proj_pred, proj_targ, n


if __name__=='__main__':
    import pandas as pd
    from tqdm import trange

    # import predefined states for testing
    from sample_rho import *
    from random_gen import *

    def do_full_ex_decomp(setup,savename,bell=True, eritas=True, random=True, num_random=100, N=1000):
        ''' Run example decompositions using the C setup.
        Params:
            setup: 'C' or ;'I'
            savename: name of file to save results to
            bell: whether to include bell states
            num_random: number of random states to decompose
            (optinonal) num_random: number of random states to decompose
        '''
        # initilize dataframe to store results
        decomp_df = pd.DataFrame({'state':[], 'n':[], 'setup': [], 'simple': [], 'angles':[], 'fidelity':[], 'projH&V_pred':[], 'projD&A_pred':[], 'projR&L_pred':[], 'projH&V_targ':[], 'projD&A_targ':[], 'projR&L_targ':[]})
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
            for setup in ['C','I']:
                for simple in [True,False]:
                    try:
                        angles, fidelity, proj_pred, proj_targ, n = jones_decompose(states[i], setup=setup,simple=simple, N=N, verbose=True)
                        decomp_df = pd.concat([decomp_df, pd.DataFrame.from_records([{'state':states_names[i], 'n':n, 'setup': setup, 'simple': simple, 'angles':angles, 'fidelity':fidelity, 'projH&V_pred':proj_pred[:4], 'projD&A_pred':proj_pred[4:8], 'projR&L_pred':proj_pred[8:], 'projH&V_targ':proj_targ[:4], 'projD&A_targ':proj_targ[4:8], 'projR&L_targ':proj_targ[8:]}])])
                    except: # if there's an error, save what we have
                        print('error! saving what we have')
                        decomp_df.to_csv(savename+'.csv')

        decomp_df.to_csv(savename+'.csv')
        return decomp_df
    
    
    # setup=input('Enter setup (C or I): ')
    # while setup != 'C' and setup != 'I':
    #     setup=input('Enter setup (C or I): ')
    bell = bool(int(input('include bell states?')))
    eritas = bool(int(input('include eritas states?')))
    random = bool(int(input('include random states?')))
    special = input('sepical name to append to file?')
    savename = f'decomp_all_{bell}_{eritas}_{random}_{special}'
    print('all', bell, eritas, random)

    do_full_ex_decomp('I',bell=bell, eritas=eritas, random=random, savename=savename)

    # jones_decompose(get_random_simplex()[0], setup = 'I', simple=False, eps_max=.99, N = 30000, verbose=True)
        
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

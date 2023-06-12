# file for jones matrix computations

# main package imports #
import numpy as np
from scipy.optimize import minimize

# for adaptive sampling #
# import tensorflow as tf
import itertools

# special methods for density matrices #
from rho_methods import *
from random_gen import *

#### jones matrices ####
def R(alpha): 
    ''' Rotation matrix for angle alpha'''
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): 
    ''' HWP matrix for angle theta'''
    return np.array([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): 
    ''' QWP matrix for angle alpha'''
    return R(alpha) @ np.array(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def get_QP(phi): 
    ''' QP matrix for angle phi'''
    return np.array(np.diag([1, np.e**(phi*1j)]))
### BBO matrix ###
BBO = np.array([[0, 0, 0, 1], [1, 0,0,0]]).T
def init_state_bg(beta, gamma): 
    ''' Initial polarization state matrix for angles beta, gamma: |psi> = cos(beta)|H> + e^(gamma*i)sin(beta)|V>'''
    return np.array([[np.cos(2*beta),0],[0,np.e**(gamma*1j)*np.sin(2*beta)]])
# initial state for simple=False
Is0 = init_state_bg(0,0)

def get_Jrho(angles, setup = 'C', simple=True, check=False):
    ''' Main function to get density matrix using Jones Matrix setup.
    Params:
        angles: list of angles for setup. see the conditionals for specifics
        setup: either 'C' for current setup or 'I' for ideal setup
        simple: [CURRENTLY DEPRICATED] boolean for whether to start with arbitrary cos(beta)|0> + sin(beta) e^(i*gamma)|1> or to use a combination of HWP and QP or HWP and 2x QP for C and I respectively
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
            rho = P @ adjoint(P)

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
            rho = P @ adjoint(P)

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
            rho = P @ adjoint(P)

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

            rho = P @ adjoint(P)

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



def get_random_Jangles(setup, simple=False):
    ''' Returns random angles for the Jrho_C setup. Confirms that the density matrix is valid.
    params: setup: either 'C' for current setup or 'I' for ideal setup
            simple: boolean for whether to start with arbitrary cos(beta)|0> + sin(beta) e^(i*gamma)|1> or to use a combination of HWP and QP or HWP and 2x QP
    '''

    def get_angles():


        if simple:
             # init state
            beta = np.random.rand()*np.pi/4 
            gamma = np.random.rand()*np.pi/2

            if setup=='C':
                # HWP
                theta = np.random.rand()*np.pi/4
                # QWPs
                alpha_ls = np.random.rand(2)*np.pi/2
                angles = [beta, gamma, theta, *alpha_ls]

            elif setup=='I':
                # HWPs
                theta_ls = np.random.rand(2)*np.pi/4
                # QWPs
                alpha_ls = np.random.rand(4)*np.pi/2
                angles = [beta, gamma, *theta_ls, *alpha_ls]
            else:
                raise ValueError(f'Invalid setup. You have {setup} but needs to be either "C" or "I".')
        else:
            if setup=='C':
                # UV HWP
                theta_UV = np.random.rand()*np.pi/4
                # QP
                phi = np.random.rand()*0.69
                # B HWP
                theta_B = np.random.rand()*np.pi/4
                # QWPs
                alpha_ls = np.random.rand(2)*np.pi/2
                angles= [theta_UV, phi, theta_B, *alpha_ls]
            elif setup=='I':
                # UVHWP 
                theta_0 = np.random.rand()*np.pi/4
                # initial QWPs
                Q_init = np.random.rand(2)*np.pi/2
                # HWPs
                theta_ls = np.random.rand(2)*np.pi/4
                # B and A QWPs
                alpha_ls = np.random.rand(4)*np.pi/2

                angles= [theta_0, *Q_init, *theta_ls, *alpha_ls]
            else:
                raise ValueError(f'Invalid setup. You have {setup} but needs to be either "C" or "I".')

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

def jones_decompose(targ_rho, targ_name='Test', setup = 'C', adapt=False, frac = 0.1, simple=False, epsilon=0.999, N = 1000, verbose=False, debug=False):
    ''' Function to decompose a given density matrix into jones matrices
    params:
        targ_rho: target density matrix
        targ_name: name of target state
        setup: either 'C' for current setup or 'I' for ideal setup
        adapt: whether to adaptively optimize the angles via gradient descent
        frac: what percentage of the domain of the angles to change each time in adapt
        simple: boolean for whether to start with arbitrary cos(beta)|0> + sin(beta) e^(i*gamma)|1> or to use a combination of HWP and QP or HWP and 2x QP
        epsilon: maximum tolerance for fidelity; if reached, halleljuah and break early!
        N: max number of times to try to optimize
        verbose: whether to include print statements.
        debug: whether to enforce try/excpet to block errors
    returns:
        angles: list of angles matching setup return. note beta and gamma, which set the initial polarization state, are the first two elements
        fidelity: fidelity of the resulting guessed state
    note: if you use Cntrl-C to break out of the function, it will return the best guess so far
    '''

    def decompose():
        func = lambda angles: get_Jrho(angles=angles, setup=setup, simple=simple)

        # initial guesses (PhiP)
        if setup=='C':
            if simple: # beta, gamma, theta, alpha1, alpha2
                bounds = [(0, np.pi/2), (0, np.pi/2), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2)]
            else: # theta_0, phi, theta_B, alpha_B1, alpha_B2
                bounds = [(0, np.pi/4), (0, 0.69), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2)]
        elif setup=='I':
            if simple: # beta, gamma, theta1, theta2, alpha1, alpha2
                bounds = [(0, np.pi/2), (0, np.pi/2), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2)]
            else: # theta_uv, alpha_0, alpha_1, theta_B, theta_A, alpha_B1, alpha_B2, alpha_A1, alpha_A2
                bounds = [(0, np.pi/4),  (0, np.pi/2), (0, np.pi/2), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2)]
        else:
            raise ValueError(f'Invalid setup. Must be either "C" or "I". You have {setup}.')

        def loss_fidelity(angles):
            ''' Function to quantify the distance between the targ and pred density matrices'''
            pred_rho = func(angles)
            fidelity = get_fidelity(pred_rho, targ_rho)
            
            return 1-np.sqrt(fidelity)

        # def wrapped_loss_fidelity(inputs):
        #     ''' Wrapped version of loss_fideliy for tensorflow tensor'''
        #     return tf.numpy_function(loss_fidelity, [inputs], tf.float32)

        # get initial result
        def minimize_angles(x0):
            result = minimize(loss_fidelity, x0=x0, bounds=bounds)
            best_angles = result.x
            rho = func(best_angles)
            fidelity = get_fidelity(rho, targ_rho)
            return best_angles, fidelity, rho

        x0 = get_random_Jangles(setup=setup, simple=simple)
        best_angles, fidelity, rho = minimize_angles(x0)
        
        # iterate eiher until we hit max bound or we get a valid rho with fidelity above the min
        if is_valid_rho(rho):
            max_best_fidelity = fidelity
            max_best_angles = best_angles
        else:
            max_best_fidelity = 0
            max_best_angles = [0 for _ in best_angles]
        n=0 # keep track of overall number of iterations
        m = 0 # keep track of number of iterations current adapt search
        while n < N and fidelity<epsilon:
            try:
                if verbose: 
                    print('n', n)
                    print(fidelity, max_best_fidelity)
                
                ## start with different initial guesses ##
                if adapt and not(np.all(max_best_angles == np.zeros(len(max_best_angles)))): # implement gradient descent
                    m+=1
                    if m>0:

                        if m==frac*N:
                            x0 = get_random_Jangles(setup=setup, simple=simple)
                            m=-frac*N

                        # angle_lines = [np.linspace(max(bounds[i][0], max_best_angles[i]*(1-m*((bounds[i][1] - bounds[i][0]) / (.1*N)))), min(max_best_angles[i]*1.1, bounds[i][1]), 10) for i in range(len(max_best_angles))]

                        x0 = [np.random.uniform(max(bounds[i][0], max_best_angles[i] * (1 - m * ((bounds[i][1] - bounds[i][0]) / (frac * N)))),
                            min(bounds[i][1], max_best_angles[i] * (1 + m * ((bounds[i][1] - bounds[i][0]) / (frac * N)))))
        for i in range(len(max_best_angles))]

                    # angle_mesh = list(itertools.product(*angle_lines))
                    # L = loss_fidelity(angle_mesh)
    
                    # grad = np.gradient(L, *angle_lines)
                    # print(grad)
                    else: # cool down
                        x0 = get_random_Jangles(setup=setup, simple=simple)



# print('gradient descent')
                    # # max_best_angles.reshape(-1, len(max_best_angles))
                    # print(max_best_angles.shape, max_best_angles[4])
                    # print(setup)
                    # print(simple)
                    # rho = func(max_best_angles)
                    # print(rho)
                    # print(loss_fidelity(max_best_angles))
                    # grad = np.gradient(loss_fidelity(max_best_angles))
                    # print(grad)
                    # max_best_angles = list(max_best_angles)
                    # print(max_best_angles)
                    # print(type(max_best_angles))
                    # x = torch.tensor(max_best_angles, requires_grad=True)
                    # loss = loss_fidelity(x)
                    # loss.backward()
                    # gradient = x.grad.detach().numpy()
                    # print(gradient)

                    # define inputs
                    # tf_vars = [tf.Variable(i) for i in np.arange(0, len(max_best_angles), 1.0)]
                    # print(tf_vars)
                    # with tf.GradientTape() as t:
                    #     output = wrapped_loss_fidelity(tf_vars)

                    # gradients = t.gradient(output, tf_vars)

                    # get n-d grid of angles
                    # angle_grid = np.meshgrid(*[np.linspace(max(bounds[i][0], max_best_angles[i]*.9), min(max_best_angles[i]*1.1, bounds[i][1]), 10) for i in range(len(max_best_angles))])
                    # grad = np.gradient(loss_fidelity(angle_grid), angle_grid)
                    # print(grad)

                    # inputs = []
                    # grids = []
                    # for j in range(len(max_best_angles)):
                    #     x = np.linspace(max(bounds[j][0], max_best_angles[j]*.9), min(max_best_angles[j]*1.1, bounds[j][1]), 10)
                    #     inputs.append(x)
                    #     grids.extend(np.meshgrid(*inputs))

                    # x= tf.convert_to_tensor(max_best_angles, dtype=tf.complex64)
                    # with tf.GradientTape() as t:
                    #     t.watch(x)
                    #     loss = tf.py_function(loss_fidelity, [x], Tout=tf.float64)
                    # gradient = t.gradient(loss, x)

                    
                    # x0 -= lr*grad
                    # print(x0)
                else: # stick with random guesses
                    x0 = get_random_Jangles(setup=setup, simple=simple)

                best_angles, fidelity, rho = minimize_angles(x0)

                if fidelity > max_best_fidelity:
                    max_best_fidelity = fidelity
                    max_best_angles = best_angles

                n+=1

            except KeyboardInterrupt:
                print('interrupted...saving best result so far')
                break

        # compute projections in 12 basis states
        proj_pred = get_all_projections(func(max_best_angles))
        proj_targ = get_all_projections(targ_rho)

        # if verbose:
        print('actual state', targ_rho)
        print('predicted state', func(max_best_angles) )
        print('num iterations', n)
        print('fidelity', max_best_fidelity)
        print('projections of predicted', proj_pred)
        print('projections of actual', proj_targ)

        return targ_name, setup, simple, n, max_best_fidelity, max_best_angles, proj_pred[:4], proj_targ[:4], proj_pred[4:8], proj_targ[4:8], proj_pred[8:], proj_targ[8:]

    if not(debug):
        try:
            return decompose()
        except Exception as e:
            print('Error!', e)
    else:
        return decompose()

if __name__=='__main__':
    # for loading data
    import pandas as pd
    # for progress bar
    from tqdm import trange
    # for parallel processing
    from multiprocessing import cpu_count, Pool

    # import predefined states for testing
    from sample_rho import *
    from random_gen import *

    def do_full_ex_decomp(savename,bell=False, eritas=False, random=False, jones_C=False, jones_I=False, roik=False, num_random=100):
        ''' Run example decompositions using the C setup.
        Params:
            setup: 'C' or ;'I'
            savename: name of file to save results to
            bell: whether to include bell states
            num_random: number of random states to decompose
            (optional) num_random: number of random states to decompose
        '''
        
        states = []
        states_names = []

        ## compile states ##
        if bell:
            for i in range(num_random):
                states_bell = [PhiP, PhiM, PsiP, PsiM]
                states_bell_names = ['PhiP', 'PhiM', 'PsiP', 'PsiM']
                states += states_bell
                states_names += states_bell_names
        print('ehre')
        print(len(states))
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
        print('hi')
        if random:
            states_RS_all = [get_random_simplex() for i in range(num_random)]
            states_RS = [states_RS_all[i][0] for i in range(num_random)]
            states_RS_names = [f'RS_{states_RS_all[i][1][0]}_{states_RS_all[i][1][1]}_{states_RS_all[i][1][2]}_{states_RS_all[i][1][3]}__{states_RS_all[i][1][4]}__{states_RS_all[i][1][5]}__{states_RS_all[i][1][6]}' for i in range(num_random)]
       
            states+=states_RS
            states_names+=states_RS_names
        print('hi2')
        if jones_C:
            # get random Jones states
            states_jones_all = [get_random_jones(setup='C', simple=False) for i in range(num_random)]
            states_jones = [states_jones_all[i][0] for i in range(num_random)]
            states_jones_names = [f'jones_{states_jones_all[i][1]}' for i in range(num_random)]

            states+=states_jones
            states_names+=states_jones_names
        print('hi3')
        if jones_I:
            # get random Jones states
            states_jones_all = [get_random_jones(setup='I', simple=False) for i in range(num_random)]
            states_jones = [states_jones_all[i][0] for i in range(num_random)]
            states_jones_names = [f'jones_{states_jones_all[i][1]}' for i in range(num_random)]

            states+=states_jones
            states_names+=states_jones_names
        print('hi4')
        print(roik)
        if roik:
            states_roik_all = [get_random_roik(.95) for i in range(num_random)]
            states_roik = [states_roik_all[i][0] for i in range(num_random)]
            states_roik_names = [f'roik_{states_roik_all[i][1]}' for i in range(num_random)]

            states+=states_roik
            states_names+=states_roik_names
        
        print('here')
        print('num states:', len(states))
        # list to hold the parameters (i, setup, simple) to pass to the pool multiprocessing object
        decomp_ls = []
        for i in trange(len(states)):
            for setup in ['C','I']:
                if bell:
                    for adapt in [True,False]:
                        decomp_ls.append((i, setup, adapt))
                else:
                    decomp_ls.append((i, setup, False))
        
        print(decomp_ls)

        ## build multiprocessing pool ##
        pool = Pool(cpu_count())

        inputs = [(states[decomp_in[0]], states_names[decomp_in[0]], decomp_in[1], decomp_in[2]) for  decomp_in in decomp_ls]        
        results = pool.starmap_async(jones_decompose, inputs).get()

        # end multiprocessing
        pool.close()
        pool.join()

        # filter None results out
        results = [result for result in results if result is not None] 

        ## save to df ##
        columns = ['state', 'setup', 'simple', 'n', 'fidelity', 'angles', 'projH&V_pred', 'projH&V_targ', 'projD&A_pred',  'projD&A_targ','projR&L_pred', 'projR&L_targ']
        decomp_df = pd.DataFrame.from_records(results, columns=columns)
        decomp_df.to_csv(join('decomp', savename+'.csv'))


    
    
    
    # setup=input('Enter setup (C or I): ')
    # while setup != 'C' and setup != 'I':
    #     setup=input('Enter setup (C or I): ')
    bell = bool(int(input('include bell states?')))
    eritas = bool(int(input('include eritas states?')))
    random = bool(int(input('include random states?')))
    jones_C = bool(int(input('include jones states in C setup?')))
    jones_I = bool(int(input('include jones states in I setup?')))
    roik = bool(int(input('include roik states?')))
    special = input('special name to append to file?')
    savename = f'decomp_all_{bell}_{eritas}_{random}_{special}'
    print('all', bell, eritas, random, jones_C, jones_I, roik)

    do_full_ex_decomp(bell=bell, eritas=eritas, random=random, jones_C = jones_C, jones_I = jones_I, roik=roik, savename=savename)

    # jones_decompose(PhiM, adapt=False, frac=.07, verbose=True, debug=True)



## ex_decomp old code ##
# initilize dataframe to store results
        # global decomp_df
        # decomp_df = pd.DataFrame({'state':[], 'n':[], 'setup': [], 'simple': [], 'angles':[], 'fidelity':[], 'projH&V_pred':[], 'projD&A_pred':[], 'projR&L_pred':[], 'projH&V_targ':[], 'projD&A_targ':[], 'projR&L_targ':[]})

        # try:
        #     angles, fidelity, proj_pred, proj_targ, n = jones_decompose(states[i], setup=setup,simple=simple, N=N, verbose=True)
        #     decomp_df = pd.concat([decomp_df, pd.DataFrame.from_records([{'state':states_names[i], 'n':n, 'setup': setup, 'simple': simple, 'angles':angles, 'fidelity':fidelity, 'projH&V_pred':proj_pred[:4], 'projD&A_pred':proj_pred[4:8], 'projR&L_pred':proj_pred[8:], 'projH&V_targ':proj_targ[:4], 'projD&A_targ':proj_targ[4:8], 'projR&L_targ':proj_targ[8:]}])])
        # except: # if there's an error, save what we have
        #     print('error! saving what we have')
        #     decomp_df.to_csv(savename+'.csv')

        # decomp_df.to_csv(savename+'.csv')
        # return decomp_df









    # jones_decompose(get_random_simplex()[0], setup = 'I', simple=False, epsilon=.99, N = 30000, verbose=True)
        
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

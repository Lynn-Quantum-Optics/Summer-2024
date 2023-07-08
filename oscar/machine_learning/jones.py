# file for jones matrix computations

# main package imports #
from os.path import isdir, join
import os
import numpy as np
from scipy.optimize import minimize, approx_fprime

# special methods for density matrices #
from rho_methods import *
from random_gen import *

### experimental variables to reflect non-idealities in model ###
# QP #
# l_p = 403 * 10**(-9) # wavelength of pump in m
# n_e_p_qp = 1.56677 # extraordinary refractive index for 403 nm for QP
# n_o_p_qp = 1.55722 # ordinary refractive index for 403 nm for QP
# d_QP = 0.5 * 10**(-3) # thickness of QP in m
# def phi_H_QP(phi_a): return (2*np.pi * d_QP * n_o_p_qp**2) / (l_p*np.sqrt(n_o_p_qp**2 - np.sin(phi_a)**2)) # phase shift for H polarization; note phi_a is the actual angle of the QP!
# def phi_V_QP(phi_a): return (2*np.pi * d_QP * n_e_p_qp**2) / (l_p*np.sqrt(n_e_p_qp**2 - np.sin(phi_a)**2)) # phase shift for V polarization; note phi_a is the actual angle of the QP!

# # PCC #
# d_PCC = 5.68 * 10**(-3) # thickness of PCC in m
# phi_H_PCC = (2*np.pi * d_PCC * n_o_p_qp) / l_p # phase shift for H polarization
# phi_V_PCC = (2*np.pi * d_PCC * n_e_p_qp) / l_p # phase shift for V polarization

# BBO #
# a = 0.9143592035643862 # ratio of max HH / max VV; determined by sweeping UV_HWP
# d_BB0 = 0.5 * 10**(-3) # thickness of BBO in m
# def n_e_bbo(l):
#     ''' extraordinary refractive index of BBO crystal for wavelength l in m'''
#     l *= 10**(-9) # convert to m
#     return 2.7359 + (0.01878*10**(-12)) / (l**2 - 0.01667*10**(-12)) - (0.01354 * l**2) / (10**(-12))
# def n_o_bbo(l):
#     ''' ordinary refractive index of BBO crystal for wavelength l in m'''
#     l *= 10**(-9) # convert to m
#     return 2.3753 + (0.01224*10**(-12)) / (l**2 - 0.0156*10**(-12)) - (0.0044 * l**2) / (10**(-12))
# def n_eff_bbo(l, gamma):
#     ''' effective refractive index of BBO crystal for wavelength l in m at angle gamma'''
#     l *= 10**(-9) # convert to m
#     return ((np.cos(gamma) / n_o_bbo(l))**2 + (np.sin(gamma) / n_e_bbo(l))**2)**(-1/2)
# def phi_BBO(gamma):
#     '''Phase shift of BBO crystal'''
#     return 2*np.pi*d_BB0*(n_eff_bbo(806, gamma) / (806*10**(-9)) - n_o_bbo(403) / (403*10**(-9)))    

#### jones matrices ####
def R(alpha): 
    ''' Rotation matrix for angle alpha'''
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): 
    ''' HWP matrix for angle theta'''
    # return np.array([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
    return R(theta) @ np.diag([-1, 1]) @ R(-theta)
def Q(alpha): 
    ''' QWP matrix for angle alpha'''
    return R(alpha) @ np.array(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def get_QP(phi): 
    ''' QP matrix for angle phi'''
    return np.array(np.diag([1, np.e**(phi*1j)]))
BBO = np.array([[0, 0, 0, 1], [1, 0,0,0]]).T
s0 = np.array([[1], [0]]) # initial |0> state

# -- experimental components -- #

# experimental BBO matrix
# full correction
# def get_BBO_expt(phi)(gamma): return np.array([[0, 0, 0, a], [np.exp(1j*phi_BBO(gamma)), 0,0,0]], dtype='complex').T 
# partial correction

# VV / HH on positive QP angles
# def a(QP_rot): 
#     return 0.003919514737857195*np.rad2deg(QP_rot) + 1.024799049989838   
# def BBO_expt(QP_rot): 
#     return np.array([[0, 0, 0, a(QP_rot)], [1, 0,0,0]], dtype='complex').T 

# VV /(HH +VV) for negative QP angles
def a(QP_rot, params=[1.00804849e-10, 1.24045054e+02, -3.66700073e-05, -3.72126982e+02, 6.33126695e-01]):
    a, b, c, d, e = params
    QP_rot = np.rad2deg(QP_rot) # convert to degrees
    return a*QP_rot**4 + b*QP_rot**3 + c*QP_rot**2 + d*QP_rot + e
def BBO_expt(QP_rot):
    return np.array([[0, 0, 0, a(QP_rot)], [1-a(QP_rot), 0,0,0]], dtype='complex').T

# def get_BBO_expt(phi)(gamma): return np.array([[0, 0, 0, a], [1, 0,0,0]], dtype='complex').T 
# set angle of BBO
# BBO_expt(phi) = BBO
# def get_QP_expt(phi_a):
#     ''' QP matrix for angle phi_a, where phi_a is the actual angle of the QP'''
#     return np.diag([np.exp(1j*phi_H_QP(phi_a)), np.exp(1j*phi_V_QP(phi_a))])


# for positive angles of QP #
# def get_phi(QP_rot, params = [-6.98200712e+04, 4.33971479e+03, 4.37424378e+02, 3.41720748e+03, 1.47435257e+03, 1.42317213e+04, 1.96811356e+01, 3.49254704e+04, -2.97037399e-01, 6.98202522e+04]):
#     '''Function based on fitting to Alec's QP vs phi sweep. Assumes input is in radians and outputs in radians.'''
#     a, b, c, d, e, f, g, h, i, j = params
#     phi = a / np.cos(QP_rot) + b*QP_rot**8 + c*QP_rot**7 +d*QP_rot**6 + e*QP_rot**5 + f*QP_rot**4 + g*QP_rot**3 + h*QP_rot**2 + i*QP_rot + j
#     # print('QP rot in deg', np.rad2deg(QP_rot))
#     # print('phi in deg', np.rad2deg(phi))
#     return -phi

# for negative angles of QP #
# bound is 0 to -0.6363 rad= -36.5 deg
def get_phi(QP_rot, params=[-1.05971375e+04,5.25511676e+03,1.18147650e+04, 1.27410922e+04, 6.07139506e+03, 3.81757342e+03, 1.94716763e+02, 5.31934589e+03, -8.01250263e-01, 1.05975049e+04]):
    '''Using my sweep data calculated using Alec's phi expression. Assumes input is in radians and outputs in radians.'''

    a, b, c, d, e, f, g, h, i, j = params
    phi = a / np.cos(QP_rot) + b*QP_rot**8 + c*QP_rot**7 +d*QP_rot**6 + e*QP_rot**5 + f*QP_rot**4 + g*QP_rot**3 + h*QP_rot**2 + i*QP_rot + j

    return -phi

def get_QP_rot(phi):
    '''Function to return inverse of get_phi, that is what the theoretical angle is'.
    Assumes input is in degrees.
    '''
    phi = np.deg2rad(phi)
    def diff(QP_rot):
        return phi - get_phi(QP_rot)
    val = minimize(diff, x0=[np.pi], bounds = [(0, 2*np.pi)]).fun[0] % (2*np.pi)
    return np.rad2deg(val)

def get_QP_expt(QP_rot):
    ''' QP matrix for angle phi_a, where phi_a is the actual angle of the QP'''
    return np.diag([1, np.exp(1j*get_phi(QP_rot))])

# experimental PCC matrix, rotated by angle beta
# def get_PCC_expt(beta=4.005): return R(beta) @ np.diag([np.exp(1j*phi_H_PCC), np.exp(1j*phi_V_PCC)])@ R(-beta) 

#### calculations ####

def get_Jrho(angles, setup = 'C0', add_noise = False, p = 0.04, expt = True, check=False):
    ''' Main function to get density matrix using Jones Matrix setup.
    Params:
        angles: list of angles for setup. see the conditionals for specifics
        setup: 'C0', 'C1', 'C2', 'I' 
        BBO_corr: 0 to not correct, 1 to partial, 2 to full correction
        gamma: angle of BBO crystal for corection; determined by maximizing fidelity for PhiP setup; from Ivy's: 0.2735921530153273
        check: boolean to check if density matrix is valid; don't call when minimizing bc throwing an error will distrupt minimzation process. the check is handled in the while statement in jones_decomp -- for this reason, set to False by default
        gamma: angle of BBO crystal; determined by maximizing fidelity for PhiP setup; from Ivy's:29.5 deg
        add_noise: boolean to add noise to the density matrix to simulate mixed states in the lab
        p: probability of noise
        expt: boolean to use experimental components or not
    '''

    if setup=='C0':
        ''' Jones matrix with  current setup, just adding one QWP on Alice. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO          -> A_Detectors
                            -> B_HWP -> B_Detectors
        '''
        
        ''' Starts with input state with beta = gamma = 0. Input angles:
            UV_HWP, QP, Bob's HWP, Bob's QWP, Alice's QWP
        '''
         # UVHWP 
        theta0 = angles[0]
        # QP
        phi = angles[1]
        # B HWP
        B_theta = angles[2]

        if expt:
            # experimental QP #
            QP = get_QP_expt(phi) # combine QP and PCC

        else:
           QP = get_QP(phi)
    
        H_UV = H(theta0)
        H_B = H(B_theta)

        if expt:
            U = np.kron(np.eye(2), H_B) @ BBO_expt(phi) @ QP @ H_UV
        else:
            U = np.kron(np.eye(2), H_B) @ BBO @ QP @ H_UV

    elif setup=='C1':
        ''' Jones matrix with adding one QWP on Bob. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO                   -> A_Detectors
                            -> B_HWP -> B_QWP -> B_Detectors
        '''
        
        ''' Starts with input state with beta = gamma = 0. Input angles:
            UV_HWP, QP, Bob's HWP, Bob's QWP, Alice's QWP
        '''
         # UVHWP 
        theta0 = angles[0]
        # QP
        phi = angles[1]
        # B HWP
        B_theta = angles[2]
        B_alpha = angles[3]

        if expt:
            # experimental QP #
            QP = get_QP_expt(phi) # combine QP and PCC
            
        else:
           QP = get_QP(phi)
    
        H_UV = H(theta0)
        H_B = H(B_theta)
        Q_B = Q(B_alpha)

        if expt:
            U = np.kron(np.eye(2), Q_B @ H_B) @ BBO_expt(phi) @ QP @ H_UV
        else:
            U = np.kron(np.eye(2), Q_B @ H_B) @ BBO @ QP @ H_UV


    elif setup == 'C2':
        ''' Jones matrix with current setup, adding QWP on Bob and Alice. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_Detectors
        '''
        
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
        if expt:
            QP = get_QP_expt(phi)
            
        else:
            QP = get_QP(phi)
        H_B = H(B_theta)
        Q_B = Q(B_alpha)
        Q_A = Q(A_alpha)
        
        if expt:
            U = np.kron(Q_A, Q_B @ H_B) @ BBO_expt(phi) @ QP @ H_UV
        else:
            U = np.kron(Q_A, Q_B @ H_B) @ BBO @ QP @ H_UV

    elif setup == 'I':
        ''' Jones matrix method with Ideal setup. Computes the rotated polarization state using the following setup:
            UV_HWP -> QP -> BBO -> A_HWP -> A_QWP -> A_QWP -> A_Detectors
                                -> B_HWP -> B_QWP -> B_QWP -> B_Detectors
        '''
        
        
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

        if expt:
            U = np.kron(Q_A1 @ Q_A2 @ H_A, Q_B1 @ Q_B2 @ H_B) @ BBO  @ Q1 @ Q0 @ H_UV
        else:
            U = np.kron(Q_A1 @ Q_A2 @ H_A, Q_B1 @ Q_B2 @ H_B) @ BBO @ Q1 @ Q0 @ H_UV

    else:
        raise ValueError(f'Invalid setup. You have {setup} but needs to be either "C0", "C1", "C2", or "I".')


    # apply unitary
    P = U @ s0
    rho = P @ adjoint(P)
    rho/= np.trace(rho)

    ## return density matrix ##
    if check:
        if is_valid_rho(rho):
            if not(add_noise):
                return rho
            else:
                rho  = (1-p)*rho + p*np.eye(4)/4
                return rho
        else:
            print(rho)
            raise ValueError('Invalid density matrix')

    else:
        if not(add_noise):
                return rho
        else:
            rho  = (1-p)*rho + p*np.eye(4)/4
            return rho


def get_random_Jangles(setup='C1', expt=True):
    ''' Returns random angles for the Jrho_C setup. Confirms that the density matrix is valid.
    params: 
        setup: see get_Jrho
        bad: boolean for whether to return U*Is_0 or U*Is_0 @ adjoint(U*Is_0) -- set to False by default.

    '''

    def get_angles():

        if setup=='C0':
            # UV HWP
            theta_UV = np.random.rand()*np.pi/4
            # QP
            if expt: 
                # phi = np.random.uniform(0, np.deg2rad(38.299))
                phi = np.random.uniform(-.6363, 0)
            else:
                phi = np.random.rand()*2*np.pi
            # B HWP
            theta_B = np.random.rand()*np.pi/4
            
            angles= [theta_UV, phi, theta_B]

        elif setup=='C1':
            # UV HWP
            theta_UV = np.random.rand()*np.pi/4
            # QP
            if expt:
                # phi = np.random.uniform(0, np.deg2rad(38.299)) # 40 degrees is the max angle for the QP
                phi = np.random.uniform(-.6363, 0)
            else:
                phi = np.random.rand()*2*np.pi
            # B HWP
            theta_B = np.random.rand()*np.pi/4
            # QWPs
            alpha = np.random.rand()*np.pi/2
            angles= [theta_UV, phi, theta_B, alpha]
            
        
        elif setup=='C2':
            # UV HWP
            theta_UV = np.random.rand()*np.pi/4
            # QP
            if expt:
                # phi =np.random.uniform(0, np.deg2rad(38.299)) # 40 degrees is the max angle for the QP
                phi = np.random.uniform(-.6363, 0)

            else:
                phi = np.random.rand()*2*np.pi
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
            raise ValueError(f'Invalid setup. You have {setup} but needs to be either "C" or "I" or "Ca".')

        return angles

    # confirm angles are valid #
    angles = get_angles()
    while not(is_valid_rho(get_Jrho(angles=angles, setup=setup))):
        angles = get_angles()
    # while np.isclose(get_Jrho(angles=angles, setup=setup) , np.zeros):
    #     angles = get_angles()
    return angles


def get_random_jones(setup='C1', expt=True, return_params=False):
    ''' Computes random angles in the ranges specified and generates the resulting states'''
    # get random angles
    angles = get_random_Jangles(setup=setup, expt=expt)
    # get density matrix
    rho = get_Jrho(setup=setup, expt=expt, angles=angles)

    if return_params: return [rho, angles]
    else: return rho

def jones_decompose(targ_rho, targ_name='Test', setup = 'C0', adapt=0, debug=False, frac = 0.001, zeta = 0.01, expt=True, gd_tune=False, save_rho = False, verbose=True, epsilon=0.999, N = 1000, add_noise=False):
    ''' Function to decompose a given density matrix into jones matrices
    params:
        targ_rho: target density matrix
        targ_name: name of target state
        setup: see get_Jrho
        adapt: 0: random hop, 1: random fan, 2: gradient descent
        frac: what percentage of the domain of the angles to change each time in adapt
        zeta: learning rate for gradient descent
        expt: whether to use experimental components
        state_num: number of the state to decompose for progress tracking
        gd_tune: whether to output parameters for gd tuning
        save_rho: whether to save the density matrix
        verbose: whether to include print statements.
        debug: whether to enforce try/excpet to block errors
        epsilon: maximum tolerance for fidelity; if reached, halleljuah and break early!
        N: max number of times to try to optimize
        add_noise: whether to add noise when trying to predict the state
        BBO_corr: what level of BBO corection to do
    returns:
        targ_name: name of target state
        setup: string representation
        adapt: 0: random hop, 1: random fan, 2: gradient descent
        frac, zeta: params for GD
        n: num of iterations until solution
        max_best_fidelity: fidelity of best guess
        max_best_angles: angle settings corresponding to best guess
        best_pred_rho: best guess at density matrix
        targ_rho: target density matrix
    '''

    # set zeta to be more aggressive for C based on tuning
    if setup=='C2':
        zeta=0.07

    elif setup=='C0':
        frac=.02
        zeta=1

    elif setup=='C1': # change these values with tuning
        frac=.02
        zeta=1

    def decompose():
        func = lambda angles: get_Jrho(angles=angles, setup=setup, expt=expt, add_noise=add_noise)

        # iset bounds for guesses
        H_bound = (0, np.pi/4)
        Q_bound = (0, np.pi/2)
        if expt:
            # QP_bound = (0, np.deg2rad(38.299))
            QP_bound = (-0.6363, 0)
        else:
            QP_bound = (0, 2*np.pi)
        if setup=='C0':
            # theta_uv, phi, theta_B
            bounds = [H_bound, QP_bound, H_bound]
        elif setup=='C1':
            bounds = [H_bound, QP_bound, H_bound, Q_bound]
        elif setup=='C2':
            bounds = [H_bound, QP_bound, H_bound, Q_bound, Q_bound]
        elif setup=='I':
            bounds = [H_bound, Q_bound, Q_bound, H_bound, H_bound, Q_bound, Q_bound, Q_bound, Q_bound]
        else:
            raise ValueError(f'Invalid setup. Must be either "C" or "I" or "Ca". You have {setup}.')

        def loss_fidelity(angles):
            ''' Function to quantify the distance between the targ and pred density matrices'''
            pred_rho = func(angles)
            fidelity = get_fidelity(pred_rho, targ_rho)
            
            return 1-np.sqrt(fidelity)

        # get initial result
        def minimize_angles(x0):
            result = minimize(loss_fidelity, x0=x0, bounds=bounds)
            best_angles = result.x
            rho = func(best_angles)
            fidelity = get_fidelity(rho, targ_rho)
            return best_angles, fidelity, rho

        x0 = get_random_Jangles(setup=setup, expt=expt)
        best_angles, fidelity, rho = minimize_angles(x0)
        
        # iterate eiher until we hit max bound or we get a valid rho with fidelity above the min
        if is_valid_rho(rho) and (fidelity < 1 or np.isclose(fidelity, 1, rtol=1e-9)):
            max_best_fidelity = fidelity
            max_best_angles = best_angles
        else:
            max_best_fidelity = 0
            max_best_angles = [0 for _ in best_angles]
        # max_best_fidelity = fidelity
        # max_best_angles = best_angles
        grad_angles = max_best_angles # keep track of the angles for gradient descent

        n=0 # keep track of overall number of iterations
        index_since_improvement = 0 # keep track of number of iterations since the max fidelity last improved
        while n < N and max_best_fidelity<epsilon:
            try:
                if verbose: 
                    print('n', n)
                    print(fidelity, max_best_fidelity)
                
                ### different strategies ###
                ## random fan ##
                if adapt==1 and not(np.all(max_best_angles == np.zeros(len(max_best_angles)))):
                    m+=1
                    if m>0:
                        if m==frac*N:
                            x0 = get_random_Jangles(setup=setup)
                            m=-frac*N

                        x0 = [np.random.uniform(max(bounds[i][0], max_best_angles[i] * (1 - m * ((bounds[i][1] - bounds[i][0]) / (frac * N)))),
                            min(bounds[i][1], max_best_angles[i] * (1 + m * ((bounds[i][1] - bounds[i][0]) / (frac * N)))))
        for i in range(len(max_best_angles))]

                    else: # cool down
                        x0 = get_random_Jangles(setup=setup)

                ## gradient descent ##
                elif adapt==2:            
                    if index_since_improvement % (frac*N)==0: # periodic random search (hop)
                        x0 = get_random_Jangles(setup=setup)
                    else:
                        gradient = approx_fprime(grad_angles, loss_fidelity, epsilon=1e-8) # epsilon is step size in finite difference
                        # if verbose: print(gradient)
                        # update angles
                        x0 = [max_best_angles[i] - zeta*gradient[i] for i in range(len(max_best_angles))]
                        grad_angles = x0


                ## random hop ##
                else:
                    x0 = get_random_Jangles(setup=setup)

                best_angles, fidelity, rho = minimize_angles(x0)

                if fidelity > max_best_fidelity and is_valid_rho(rho) and (fidelity < 1 or np.isclose(fidelity, 1, rtol=1e-9)):
                    max_best_fidelity = fidelity
                    max_best_angles = best_angles
                    index_since_improvement = 0
                elif not(is_valid_rho(rho)):
                    n-=1
                # if fidelity > max_best_fidelity and (fidelity < 1 or np.isclose(fidelity, 1, rtol=1e-9)):
                #     max_best_fidelity = fidelity
                #     max_best_angles = best_angles
                #     index_since_improvement = 0
                else:
                    index_since_improvement += 1

                n+=1

            except KeyboardInterrupt:
                print('interrupted...saving best result so far')
                break

        # compute projections in 12 basis states
        best_pred_rho = func(max_best_angles)
        proj_pred = get_all_projs(func(max_best_angles))
        proj_targ = get_all_projs(targ_rho)

        # if verbose:
        # print('index of state generated', i)
        print('actual state', targ_rho)
        print('predicted state', func(max_best_angles) )
        print('num iterations', n)
        print('fidelity', max_best_fidelity)
        print('projections of predicted', proj_pred)
        print('projections of actual', proj_targ)

        if not(gd_tune):
            if save_rho: # save best predicted and actual rho
                # create new directory
                if not(isdir(join('decomp', targ_name, setup))):
                    os.makedirs(join('decomp', targ_name, setup))
                # save rho
                np.save(join('decomp', targ_name, setup, f'pred_rho_{n}_{max_best_fidelity}'), best_pred_rho)
                np.save(join('decomp', targ_name, setup, f'targ_rho'), targ_rho)

            return targ_name,setup, adapt, n, max_best_fidelity, max_best_angles, best_pred_rho, targ_rho

        else:
            return setup, frac, zeta, n, max_best_fidelity

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
    from tqdm import tqdm, trange
    # for parallel processing
    from multiprocessing import cpu_count, Pool
    # # for viewing arguments of function
    import matplotlib.pyplot as plt
    # for using preset inputs in function
    from functools import partial

    # import predefined states for testing
    from sample_rho import *
    from random_gen import *


    def do_full_ex_decomp(setup,expt=True, adapt=0, bell=False, e0=False,e1=False, random=False, jones_C=False, jones_I=False, roik=False, num_random=100, debug=False, savename='test') :
        ''' Run example decompositions using the C setup.
        Params:
            setup: 'C' or 'I' or 'Ca' or 'A' for all.
            expt: whether to use experimental components
            adapt: 0 for no adaptation, 1 for random fan, 2 for gradient descent
            savename: name of file to save results to
            bell: whether to include bell states
            num_random: number of random states to decompose
            debug: whether to run in debug mode
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
        if e0:
            # get random Eritas states
            states_E0_all = [get_random_E0() for i in range(num_random)]
            states_E0 = [states_E0_all[i][0] for i in range(num_random)]
            states_E0_names = [f'E0_{states_E0_all[i][1][0]}_{states_E0_all[i][1][1]}' for i in range(num_random)]

            states+=states_E0
            states_names+=states_E0_names

        if e1:
            states_E1_all = [get_random_E1() for i in range(num_random)]
            states_E1 = [states_E1_all[i][0] for i in range(num_random)]   
            states_E1_names = [f'E1_{states_E1_all[i][1][0]}_{states_E1_all[i][1][1]}' for i in range(num_random)]

            states+=states_E1
            states_names+=states_E1_names
        if random:
            states_RS_all = [get_random_simplex(return_params=True) for i in range(num_random)]
            states_RS = [states_RS_all[i][0] for i in range(num_random)]
            states_RS_names = [f'RS_{states_RS_all[i][1][0]}_{states_RS_all[i][1][1]}_{states_RS_all[i][1][2]}_{states_RS_all[i][1][3]}__{states_RS_all[i][1][4]}__{states_RS_all[i][1][5]}__{states_RS_all[i][1][6]}' for i in range(num_random)]
       
            states+=states_RS
            states_names+=states_RS_names
        if jones_C:
            # get random Jones states
            states_jones_all = [get_random_jones(setup='C', return_params=True) for i in range(num_random)]
            states_jones = [states_jones_all[i][0] for i in range(num_random)]
            states_jones_names = [f'jones_{states_jones_all[i][1]}' for i in range(num_random)]

            states+=states_jones
            states_names+=states_jones_names
        if jones_I:
            # get random Jones states
            states_jones_all = [get_random_jones(setup='I', return_params=True) for i in range(num_random)]
            states_jones = [states_jones_all[i][0] for i in range(num_random)]
            states_jones_names = [f'jones_{states_jones_all[i][1]}' for i in range(num_random)]

            states+=states_jones
            states_names+=states_jones_names
        if roik:
            print('generating random separable roik states...')
            states_roik_all = []
            for i in trange(num_random):
                states_roik_all.append(get_random_hurwitz(.95))
            states_roik = [states_roik_all[i][0] for i in range(num_random)]
            states_roik_names = [f'roik_{states_roik_all[i][1]}' for i in range(num_random)]

            states+=states_roik
            states_names+=states_roik_names
        
        print('num states:', len(states))
        # list to hold the parameters (i, setup, simple) to pass to the pool multiprocessing object
        decomp_ls = []
        for i in trange(len(states)):
            if setup=='A':
                for option in ['C0', 'C1', 'C2', 'I']:
                    decomp_ls.append((i, option))
            else:
                decomp_ls.append((i, setup))
        
        print(decomp_ls)

        # get function with preset inputs
        decomp = partial(jones_decompose, adapt=adapt, debug=debug, expt=expt)

        ## build multiprocessing pool ##
        pool = Pool(cpu_count())

        inputs = [(states[decomp_in[0]], states_names[decomp_in[0]], decomp_in[1]) for  decomp_in in decomp_ls]        
        results = pool.starmap_async(decomp, inputs).get()

        ## end multiprocessing ##
        pool.close()
        pool.join()

        # filter None results out
        results = [result for result in results if result is not None] 

        ## save to df ##
        columns = ['state', 'setup', 'adapt', 'n', 'fidelity', 'angles', 'targ_rho', 'pred_rho']
        decomp_df = pd.DataFrame.from_records(results, columns=columns)
        decomp_df.to_csv(join('decomp', savename+'.csv'))

    def tune_gd(f_min=0.001, f_max = 0.1, f_it =10, zeta_min=0.001, zeta_max=1, zeta_it=10, num_to_avg=10, do_compute=False, do_plot=False):
        ''' Function to tune the gradient descent algorithm based on the PhiM state.
        Params:
            f_min: minimum fraction of N to do hop
            f_max: maximum fraction of N to do hop
            f_it: number of iterations between f_min and f_max
            zeta_min: minimum zeta (learning rate)
            zeta_max: maximum zeta (learning rate)
            zeta_it: number of iterations between zeta_min and zeta_max
            num_to_avg: number of times to repeat each config to get average and sem
            do_compute: whether to compute new data
            do_plot: whether to plot data
        N.b. defaults based on initial run.
        '''
        assert do_compute or do_plot, 'must do at least one of compute or plot'

        savename=f'gd_tune_{f_min}_{f_max}_{f_it}_{zeta_min}_{zeta_max}_{zeta_it}_{num_to_avg}_all'

        f_ls = np.logspace(np.log10(f_min), np.log10(f_max), f_it)
        zeta_ls = np.logspace(np.log10(zeta_min), np.log10(zeta_max), zeta_it)
        sfz_ls = []
        sfz_unique = []
        f_plot_ls = []
        zeta_plot_ls = []
        for frac in f_ls:
            for zeta in zeta_ls:
                f_plot_ls.append(frac)
                zeta_plot_ls.append(zeta)
                for setup in ['Ca', 'C', 'I']:
                    sfz_unique.append([setup, frac, zeta]) # get unique configs
                    for j in range(num_to_avg): # repeat each config num_to_avg times
                        sfz_ls.append([setup, frac, zeta])
                # sfz_unique.append(['Ca', frac, zeta]) # get unique configs
                # for j in range(num_to_avg): # repeat each config num_to_avg times
                #     sfz_ls.append(['Ca', frac, zeta])
        
        print(sfz_ls, len(sfz_ls))
        print(savename)
        
        def compute():

            ## build multiprocessing pool ##
            pool = Pool(cpu_count())

            # targ_rho, targ_name='Test', setup = 'C', adapt=0, frac = 0.1, zeta = 0, state_num = 0, gd_tune=False, debug

            # old: use PhiM
            # inputs = [(PhiM, 'PhiM', sfz[0], 2, sfz[1], sfz[2], True) for  sfz in sfz_ls]        

            # new: use random states
            inputs = [(get_random_simplex(), 'RS', sfz[0], 2, sfz[1], sfz[2], i, True) for i, sfz in enumerate(sfz_ls)]
            results = pool.starmap_async(jones_decompose, inputs).get()

            # end multiprocessing
            pool.close()
            pool.join()

            # filter None results out
            results = [result for result in results if result is not None]

            # get all unique configs
            cols = ['setup', 'frac', 'zeta', 'n', 'fidelity']
            df = pd.DataFrame.from_records(results, columns=cols)
            df.to_csv(join('decomp', savename+'.csv'))

        def plot():
            n_Ca_ls =[]
            n_Ca_sem_ls = []
            fidelity_Ca_ls = []
            fidelity_Ca_sem_ls = []
            n_C_ls = []
            n_C_sem_ls = []
            fidelity_C_ls = []
            fidelity_C_sem_ls = []
            n_I_ls = []
            n_I_sem_ls = []
            fidelity_I_ls = []
            fidelity_I_sem_ls = []

            for sfz in sfz_unique:
                setup = sfz[0]
                df_sfz = df.loc[(df['setup'] == setup) & np.isclose(df['frac'], sfz[1], rtol=1e-4) & np.isclose(df['zeta'], sfz[2], rtol=1e-4) & (df['n'] > 0) & ((np.isclose(df['fidelity'], 1, rtol=1e-4)) | (df['fidelity']<1))]
                df_test = df.loc[(df['setup']==setup) & np.isclose(df['frac'], sfz[1], rtol=1e-4) & np.isclose(df['zeta'], sfz[2], rtol=1e-4)]
                df_test_n = df_test.loc[df_test['n'] != 0]
                assert len(df_sfz) > 0, f'no results for this config {sfz}_{df_test_n}'

                n_avg = np.mean(df_sfz['n'].to_numpy())
                n_sem = np.std(df_sfz['n'].to_numpy())/np.sqrt(len(df_sfz['n'].to_numpy()))
                fidelity_avg = np.mean(df_sfz['fidelity'].to_numpy())
                fidelity_sem = np.std(df_sfz['fidelity'].to_numpy())/np.sqrt(len(df_sfz['fidelity'].to_numpy()))

                if setup=='Ca':
                    n_Ca_ls.append(n_avg)
                    n_Ca_sem_ls.append(n_sem)
                    fidelity_Ca_ls.append(fidelity_avg)
                    fidelity_Ca_sem_ls.append(fidelity_sem)
                elif setup=='C':
                    n_C_ls.append(n_avg)
                    n_C_sem_ls.append(n_sem)
                    fidelity_C_ls.append(fidelity_avg)
                    fidelity_C_sem_ls.append(fidelity_sem)
                elif setup=='I':
                    n_I_ls.append(n_avg)
                    n_I_sem_ls.append(n_sem)
                    fidelity_I_ls.append(fidelity_avg)
                    fidelity_I_sem_ls.append(fidelity_sem)

            # save csv summary of results #
            summary = pd.DataFrame()
            summary['frac'] = f_plot_ls
            summary['zeta'] = zeta_plot_ls
            summary['n_Ca'] = n_Ca_ls
            summary['n_Ca_sem'] = n_Ca_sem_ls
            summary['fidelity_Ca'] = fidelity_Ca_ls
            summary['fidelity_Ca_sem'] = fidelity_Ca_sem_ls
            summary['n_C'] = n_C_ls
            summary['n_C_sem'] = n_C_sem_ls
            summary['fidelity_C'] = fidelity_C_ls
            summary['fidelity_C_sem'] = fidelity_C_sem_ls
            summary['n_I'] = n_I_ls
            summary['n_I_sem'] = n_I_sem_ls
            summary['fidelity_I'] = fidelity_I_ls
            summary['fidelity_I_sem'] = fidelity_I_sem_ls
            summary.to_csv(join('decomp', savename+'_summary.csv'))
            
            pd.set_option('display.max_columns', None)
            # print best configurations #
            print('best Ca config:\n', summary.sort_values(['n_Ca', 'fidelity_Ca'], ascending=[True, False]).head())
            print('best C config:\n', summary.sort_values(['n_C', 'fidelity_C'], ascending=[True, False]).head())
            print('------------')
            print('best I config:\n', summary.sort_values(['n_I', 'fidelity_I'], ascending=[True, False]).head())

            fig= plt.figure()
            ax0 = fig.add_subplot(311, projection='3d')
            sc0= ax0.scatter(f_plot_ls, zeta_plot_ls, n_Ca_ls, marker='o', c=np.array(fidelity_Ca_ls), cmap=plt.cm.viridis)
            cb0 = fig.colorbar(sc0, ax=ax0, shrink=1)
            cb0.ax.set_position(cb0.ax.get_position().translated(0.09, 0))
            ax0.set_xlabel('$f$')
            ax0.set_ylabel('$\zeta$')
            ax0.set_zlabel('$\overline{n}$')
            ax0.set_title('Ca setup')

            ax1 = fig.add_subplot(312, projection='3d')
            sc1= ax1.scatter(f_plot_ls, zeta_plot_ls, n_C_ls, marker='o', c=np.array(fidelity_C_ls), cmap=plt.cm.viridis)
            cb1 = fig.colorbar(sc1, ax=ax1, shrink=1)
            cb1.ax.set_position(cb1.ax.get_position().translated(0.09, 0))
            ax1.set_xlabel('$f$')
            ax1.set_ylabel('$\zeta$')
            ax1.set_zlabel('$\overline{n}$')
            ax1.set_title('C setup')

            ax2 = fig.add_subplot(313, projection='3d')
            sc2= ax2.scatter(f_plot_ls, zeta_plot_ls, n_I_ls, marker='o', c=np.array(fidelity_I_ls), cmap=plt.cm.viridis)
            cb2 = fig.colorbar(sc2, ax=ax2, shrink=1)
            cb2.ax.set_position(cb2.ax.get_position().translated(0.09, 0))
            ax2.set_xlabel('$f$')
            ax2.set_ylabel('$\zeta$')
            ax2.set_zlabel('$\overline{n}$')
            ax2.set_title('I setup')

            # fig.set_size_inches(6, 15)
            # plt.tight_layout()

            plt.savefig(join('decomp', savename+'.pdf'))
            plt.show()
    
        if do_compute: compute()
        if do_plot:
            df = pd.read_csv(join('decomp', savename+'.csv'))
            plot()


    resp = int(input('0 to run decomp_ex_all, 1 to tune gd, 2  to run eritas states, 3 to run bell, 4 to tune gamma for BBO_expt(phi) '))
    
    ## test states and get average fidelity ##
    if resp == 0:
        setup = input('Enter setup: C0, C1, C2, I, or A ') 
        assert setup in ['C0', 'C1', 'C2', 'I', 'A'], f'invalid setup. you have {setup} '
        adapt = int(input('0 for random hop, 1 for random fan, 2 for gradient descent '))
        assert adapt in [0, 1, 2], f'invalid adapt. you have {adapt}'
        bell = bool(int(input('include bell states? ')))
        e0 = bool(int(input('include eritas 0 state? ')))
        e1 = bool(int(input('include eritas 1 state? ')))
        random = bool(int(input('include random states?')))
        jones_C = bool(int(input('include jones states in C setup? ')))
        jones_I = bool(int(input('include jones states in I setup? ')))
        roik = bool(int(input('include roik states? ')))
        special = input('special name to append to file? ')
        num_random = int(input('number of random states to generate? '))
        savename = f'decomp_all_{bell}_{e0}_{e1}_{random}_{special}'
        debug = bool(int(input('debug?')))
        print(setup, adapt, bell, e0, e1, random, jones_C, jones_I, roik, num_random, debug)

        do_full_ex_decomp(setup=setup, bell=bell, e0=e0, e1=e1,random=random, jones_C = jones_C, jones_I = jones_I, roik=roik, savename=savename, num_random=num_random, debug=debug, adapt=adapt)

    elif resp==1:
         ## optimize gradient descent params, f and zeta ##
        do_compute = bool(int(input('run computation? ')))
        do_plot = bool(int(input('plot results? ')))
        num_to_avg= int(input('number of times to average? '))
        if not(isdir('decomp')): os.makedirs('decomp')
        tune_gd(do_compute=do_compute, do_plot=do_plot, f_it=20, zeta_it=20, num_to_avg=num_to_avg)

    elif resp==2: # to generate results for experimental tests
        # import function to help make separate columns for the angles
        from itertools import zip_longest
        n = 6 # number of states to generate
        # initialize list
        # states = [PhiP, PhiM, PsiP, PsiM]
        # states_names = ['PhiM', 'PhiP', 'PsiM', 'PsiP']
        
        # for state in states:
        #     for setup in ['C0', 'C1']:
        #         eta_setup.append(None)
        #         chi_setup.append(None)

        states = []
        states_names = []
        eta_setup = []
        chi_setup = []

        # populate eritas states
        eta_ls = np.linspace(0, np.pi/4, n) # set of eta values to sample
        chi_ls = np.linspace(0, np.pi/2, n) # set of chi values to sample
        eta_fixed_ls= [np.pi/4, np.pi/3, np.pi/8] # fixed values for state generation
        chi_fixed= np.pi/3
        for eta_fixed in eta_fixed_ls:
            for chi in chi_ls:
                states.append(get_E0(eta_fixed, chi))
                states_names.append('E0')
                for setup in ['C0']:
                    eta_setup.append(np.rad2deg(eta_fixed))
                    chi_setup.append(np.rad2deg(chi))
            

        # for eta in eta_ls:
        #     states.append(get_E1(eta, np.pi/3))
        #     states_names.append('E1')
        #     for setup in ['C0', 'C1']:
        #         chi_setup.append(np.rad2deg(chi_fixed))
        #         eta_setup.append(np.rad2deg(eta))
            
        # get function with preset inputs
        decomp = partial(jones_decompose, adapt=2, debug=False, expt=True)

        decomp_ls = []
        for i in range(len(states)):
            for setup in ['C0']:
                decomp_ls.append((i, setup))

         ## build multiprocessing pool ##
        pool = Pool(cpu_count())

        inputs = [(states[decomp_in[0]], states_names[decomp_in[0]], decomp_in[1]) for  decomp_in in decomp_ls]        
        results = pool.starmap_async(decomp, inputs).get()

        ## end multiprocessing ##
        pool.close()
        pool.join()

        # filter None results out
        results = [result for result in results if result is not None] 

        ## get to df ##
        decomp_df = pd.DataFrame()
        columns = ['state', 'setup', 'adapt', 'n', 'fidelity', 'angles', 'targ_rho', 'pred_rho']
        decomp_df = pd.DataFrame.from_records(results, columns=columns)
        
        # parse df to make angles explicit and in degrees #
        angles_unpacked = list(zip_longest(*decomp_df['angles']))
        angles_unpacked= [np.array(angle) for angle in angles_unpacked]
        angles_unpacked = np.array(angles_unpacked)
        angles_unpacked = angles_unpacked.T
        angles_df_0 =  pd.DataFrame(angles_unpacked)
        print(angles_df_0)
        # split into C0 and C1
        angles_df_C0 = angles_df_0.loc[decomp_df['setup']=='C0']
        # angles_df_C0.columns = ['UV_HWP', 'QP', 'B_HWP', 'B_QWP']
        angles_df_C0.columns = ['UV_HWP', 'QP', 'B_HWP']
        # angles_df_C1 = angles_df_0.loc[decomp_df['setup']=='C1']
        # angles_df_C1.columns = ['UV_HWP', 'QP', 'B_HWP', 'B_QWP']

        # angles_df = pd.concat([angles_df_C0, angles_df_C1])
        angles_df  = angles_df_C0

        # prepare final output df #
        return_df = pd.DataFrame()
        return_df['state'] = decomp_df['state']
        return_df['eta'] = eta_setup
        return_df['chi'] = chi_setup
        return_df['setup'] = decomp_df['setup']
        return_df['UV_HWP'] = angles_df['UV_HWP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
        return_df['QP'] = angles_df['QP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
        return_df['B_HWP'] = angles_df['B_HWP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
        # return_df['B_QWP'] = angles_df['B_QWP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
        return_df['fidelity'] = decomp_df['fidelity']
        return_df['n'] = decomp_df['n']
        return_df['pred_rho'] = decomp_df['pred_rho']
        return_df['actual_rho'] = decomp_df['targ_rho']

        print('saving!')
        return_df.to_csv(join('decomp', 'ertias_2_fita2.csv'))

    elif resp==3: # to generate results for experimental tests
        # import function to help make separate columns for the angles
        from itertools import zip_longest

        states = [PhiP, PhiM, PsiP, PsiM]
        states_names = ['PhiP', 'PhiM', 'PsiP', 'PsiM']

        eps_ls = [0.999] # list of max fidelity values to sample

        for eps in eps_ls:
            # get function with preset inputs
            decomp = partial(jones_decompose, adapt=2, debug=False, expt=True, epsilon=eps, add_noise=False)

            decomp_ls = []
            for i in range(len(states)):
                for setup in ['C0', 'C1']:
                    decomp_ls.append((i, setup))

            ## build multiprocessing pool ##
            pool = Pool(cpu_count())

            inputs = [(states[decomp_in[0]], states_names[decomp_in[0]], decomp_in[1]) for  decomp_in in decomp_ls]        
            results = pool.starmap_async(decomp, inputs).get()

            ## end multiprocessing ##
            pool.close()
            pool.join()

            # filter None results out
            results = [result for result in results if result is not None] 

            ## get to df ##
            decomp_df = pd.DataFrame()
            columns = ['state', 'setup', 'adapt', 'n', 'fidelity', 'angles', 'targ_rho', 'pred_rho']
            decomp_df = pd.DataFrame.from_records(results, columns=columns)
            
            # parse df to make angles explicit and in degrees #
            angles_unpacked = list(zip_longest(*decomp_df['angles']))
            angles_unpacked= [np.array(angle) for angle in angles_unpacked]
            angles_unpacked = np.array(angles_unpacked)
            angles_unpacked = angles_unpacked.T
            angles_df_0 =  pd.DataFrame(angles_unpacked)
            print(angles_df_0)
            # split into C0 and C1
            angles_df_C0 = angles_df_0.loc[decomp_df['setup']=='C0']
            angles_df_C0.columns = ['UV_HWP', 'QP', 'B_HWP', 'B_QWP']
            angles_df_C1 = angles_df_0.loc[decomp_df['setup']=='C1']
            angles_df_C1.columns = ['UV_HWP', 'QP', 'B_HWP', 'B_QWP']

            angles_df = pd.concat([angles_df_C0, angles_df_C1])
            

            # prepare final output df #
            return_df = pd.DataFrame()
            return_df['state'] = decomp_df['state']
            return_df['setup'] = decomp_df['setup']
            return_df['UV_HWP'] = angles_df['UV_HWP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
            return_df['QP'] = angles_df['QP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
            return_df['B_HWP'] = angles_df['B_HWP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
            return_df['B_QWP'] = angles_df['B_QWP'].map(lambda x: np.rad2deg(x) if x and x is not None else None)
            return_df['fidelity'] = decomp_df['fidelity']
            return_df['n'] = decomp_df['n']
            return_df['pred_rho'] = decomp_df['pred_rho']
            return_df['targ_rho'] = decomp_df['targ_rho']

            print('saving!')
            return_df.to_csv(join('decomp', f'bell_{eps}_ra.csv'))

    elif resp==4:
        ''' Make plots treating each of the 3 components as indepndent vars'''

        def make_plots():

            fig, ax = plt.subplots(1,2, figsize=(20,10))
            UV_HWP_opt = []
            QP_opt = []
         
            
            # UV #
            UV_HWP_theta_ls = np.linspace(0, np.pi/2, 1000)
            fidelity_ls = []
            QP_theta = np.deg2rad(-24.1215)
            PCC_beta = np.deg2rad(4.005)
            B_QWP_theta = 0
            for UV_HWP_theta in UV_HWP_theta_ls:
                angles = [UV_HWP_theta, QP_theta, B_QWP_theta, PCC_beta]
                rho_calc= get_Jrho(angles, setup='C0', BBO_corr=BBO_corr)
                fidelity = get_fidelity(rho_calc, PhiP)
                fidelity_ls.append(fidelity)

            ax[0].scatter(np.rad2deg(UV_HWP_theta_ls), fidelity_ls, label='BBO_corr=%d'%(BBO_corr))
            ax[0].scatter(np.rad2deg(UV_HWP_theta_ls[np.argmax(fidelity_ls)]), max(fidelity_ls), marker='x', label='(%.3g, %.3g)'%(np.rad2deg(UV_HWP_theta_ls[np.argmax(fidelity_ls)]), max(fidelity_ls)), s=100)
            UV_HWP_opt.append(np.rad2deg(UV_HWP_theta_ls[np.argmax(fidelity_ls)]))

            # QP #
            QP_theta_ls = np.linspace(-np.deg2rad(38.75), np.deg2rad(38.75), 1000)
            fidelity_ls = []
            UV_HWP_theta = np.deg2rad(65.39980)
            PCC_beta = np.deg2rad(4.005)
            B_QWP_theta = 0
            for QP_theta in QP_theta_ls:
                angles = [UV_HWP_theta, QP_theta, B_QWP_theta, PCC_beta]
                rho_calc= get_Jrho(angles, setup='C0', BBO_corr=BBO_corr)
                fidelity = get_fidelity(rho_calc, PhiP)
                fidelity_ls.append(fidelity)

            ax[1].scatter(np.rad2deg(QP_theta_ls), fidelity_ls, label='BBO_corr=%d'%(BBO_corr))
            ax[1].scatter(np.rad2deg(QP_theta_ls[np.argmax(fidelity_ls)]), max(fidelity_ls), marker='x', label='(%.3g, %.3g)'%(np.rad2deg(QP_theta_ls[np.argmax(fidelity_ls)]), max(fidelity_ls)), s=100)
            QP_opt.append(np.rad2deg(QP_theta_ls[np.argmax(fidelity_ls)]))


            # PCC #
            PCC_beta_ls = np.linspace(0, 2*np.pi, 1000)
            fidelity_ls = []
            UV_HWP_theta = np.deg2rad(65.39980)
            QP_theta = np.deg2rad(-24.1215)
            B_QWP_theta = 0
            for PCC_beta in PCC_beta_ls:
                angles = [UV_HWP_theta, QP_theta, B_QWP_theta, PCC_beta]
                rho_calc= get_Jrho(angles, setup='C0', BBO_corr=BBO_corr)
                fidelity = get_fidelity(rho_calc, PhiP)
                fidelity_ls.append(fidelity)
            ax[2].scatter(np.rad2deg(PCC_beta_ls), fidelity_ls, label='BBO_corr=%d'%(BBO_corr))
            
            ax[2].scatter(np.rad2deg(PCC_beta_ls[np.argmax(fidelity_ls)]), max(fidelity_ls), marker='x', label='(%.3g, %.3g)'%(np.rad2deg(PCC_beta_ls[np.argmax(fidelity_ls)]), max(fidelity_ls)), s=100)
            PCC_opt.append(np.rad2deg(PCC_beta_ls[np.argmax(fidelity_ls)]))

        
        ax[0].set_title('UV sweep, $\hat{\\theta}_{UV} = 65.3998$')
        ax[1].set_title('QP sweep, $\hat{\\phi}_{QP} = -24.1215$')
        ax[2].set_title('PCC sweep,$\hat{\\beta}_{PCC} = 4.005$')

        ax[0].axvline(65.39998, color='red', linestyle='--')
        ax[1].axvline(-24.1215, color='red', linestyle='--')
        ax[2].axvline(4.005, color='red', linestyle='--')

        ax[0].axvline(np.mean(UV_HWP_opt), color='blue', linestyle='--')
        ax[1].axvline(np.mean(QP_opt), color='blue', linestyle='--')
        ax[2].axvline(np.mean(PCC_opt), color='blue', linestyle='--')

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[0].set_xlabel('$\\theta_{UV}$')
        ax[0].set_ylabel('Fidelity')
        ax[1].set_xlabel('$\\phi_{QP}$')
        ax[1].set_ylabel('Fidelity')
        ax[2].set_xlabel('$\\beta_{PCC}$')
        ax[2].set_ylabel('Fidelity')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.suptitle('Sweep of settings, $|\Phi^+\\rangle$')
        print('saving')
        plt.savefig(join('decomp', 'config_sweep.pdf'))
                
        # run through all combinations of BBO_corr, add_noise, gamma
        make_plots()

    # elif resp==5:
    #     ''' Parallel cooridinate plot of the 4 angles for PhiP and PsiP'''
    #     df = pd.read_csv(join('decomp', 'config_sweep.csv')) 

    #    elif resp==4:
    #     ''' Tune angle of BBO gamma based on PhiP state'''
    #     from scipy.optimize import curve_fit
    #     # from config
    #     UV_HWP_theta = np.deg2rad(65.39980)
    #     QP_theta = np.deg2rad(-24.1215)
    #     PCC_beta = np.deg2rad(4.005)
    #     B_QWP_theta = 0
    #     angles = [UV_HWP_theta, QP_theta, B_QWP_theta, PCC_beta]

    #     def loss_func(gamma):
    #         # reset BBO wth gamma
    #         # calc state
    #         rho_calc= get_Jrho(angles, setup='C0')
    #         fidelity = get_fidelity(rho_calc, PhiP)
    #         return -fidelity

    #     def plot_gamma():
    #         gamma_ls = np.linspace(0, np.pi/2, 1000)
    #         fidelity_ls = []
    #         for gamma in gamma_ls:
    #             rho_calc= get_Jrho(angles, setup='C0')
    #             fidelity = get_fidelity(rho_calc, PhiP)
    #             fidelity_ls.append(fidelity)
    #         print('fidelity', max(fidelity_ls))
    #         print('gamma', gamma_ls[np.argmax(fidelity_ls)])
    #         plt.figure(figsize=(12,8))
    #         plt.scatter(gamma_ls, fidelity_ls)

    #         # fit sin2
    #         # def sin2(x, a, b, c, d):
    #         #     return a*np.sin(b*x+c)**2+d

    #         # popt, pcov = curve_fit(sin2, gamma_ls, fidelity_ls)
    #         # plt.plot(gamma_ls, sin2(gamma_ls, *popt), 'r-', label='$%.3g\sin(%.3g \gamma + %.3g) + %.3g$'%tuple(popt))
    #         # plt.legend()
    #         plt.xlabel('$\gamma$')
    #         plt.ylabel('Fidelity')
    #         plt.title('Determining optimal BBO angle $\gamma$ for current $|\Phi^+\\rangle$ setup')
    #         plt.savefig(join('decomp', 'bbo_tune.pdf'))
    #         plt.show()

    #     # gamma = 0 # initial guess
    #     # loss = loss_func(gamma)
    #     # print('loss', loss)
    #     # gamma_min = minimize(loss_func, gamma).x
    #     # print(gamma_min)
    #     # rho_calc= get_Jrho(angles, setup='C0', gamma=gamma_min)
    #     rho_calc= get_Jrho(angles, setup='C0')
    #     print('rho', rho_calc)
    #     print('rho_actual', PhiP)
    #     fidelity = get_fidelity(rho_calc, PhiP)
    #     print('fidelity',fidelity)
    #     # print('gamma min is', gamma_min)

    #     # plot_gamma()      
        

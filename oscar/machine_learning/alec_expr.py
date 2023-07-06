# file to implement gradient descent on Alec's expression for adding another QWP

import numpy as np
from scipy.optimize import minimize, approx_fprime

from rho_methods import *
from sample_rho import *

def alec_expr(angles):
    '''Get density matrix based on Alec's expression for adding another QWP'''
    # unpack angles
    alpha, gamma, delta, phi = angles

    a_state = np.cos(alpha)*(np.cos(gamma)*np.cos(delta) - 1j * np.sin(gamma)*np.sin(delta)) * np.array([1, 0, 0, 0]).reshape(4,1) + np.cos(alpha)*(np.sin(gamma)*np.cos(delta) + 1j*np.cos(gamma)*np.sin(delta))*np.array([0,1,0,0]).reshape(4,1) + np.sin(alpha)*(np.cos(gamma)*np.sin(delta) + 1j*np.sin(gamma)*np.cos(delta))*np.exp(1j*phi)*np.array([0,0,1,0]).reshape(4,1) + np.sin(alpha)*(np.sin(gamma)*np.sin(delta) - 1j*np.cos(gamma)*np.cos(delta))*np.exp(1j*phi)*np.array([0,0,0,1]).reshape(4,1)

    return a_state @ a_state.conj().T

def alec_decompose(targ_rho, targ_name = 'Test',zeta=.7, frac=0.1, N = 1000, eps=0.999, verbose=False):
    '''Decomposes target state into params pased on experimental components for Alec's expression; modified from jones_decompose method in rho_methods.py
    __
    Parameters:
        targ_rho: target state to decompose, 4x4 numpy array
        name: name of state to decompose, string
        zeta: learning rate, float
        frac: how often to break the GD and get random angles, float
        N: max number of iterations, int
        eps: max fidelity, float
        verbpse: whether to print out progress, bool
    __
    Returns:
        angles: angles for each component, 1x4 numpy array
        pred_rho: predicted state from angles, 4x4 numpy array
    '''

    def get_random_angles():
        '''Get random angles for each component'''
        # alpha, gamma, delta, phi
        # return [np.random.rand()*np.pi/2, np.random.rand()*np.pi/2, np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
        return np.random.rand(4)*2*np.pi

    # set bounds for angles
    # bounds = [(0,np.pi/2),(0,np.pi/2),(0,np.pi/2),(0,2*np.pi)]
    bounds = [(0, 2*np.pi),(0, 2*np.pi),(0, 2*np.pi),(0, 2*np.pi)]

    # functions for GD #
    def loss_fidelity(x0):
        pred_rho = alec_expr(x0)
        fidelity = get_fidelity(pred_rho, targ_rho)
        return 1-np.sqrt(fidelity)

    def minimize_angles(x0):
        result = minimize(loss_fidelity, x0=x0, bounds=bounds)
        best_angles = result.x
        rho = alec_expr(best_angles)
        fidelity = get_fidelity(rho, targ_rho)
        return best_angles, fidelity, rho

    # initialize angles
    x0 = get_random_angles()
    angles, fidelity, rho = minimize_angles(x0)

    # set initial bests
    best_angles = angles
    best_fidelity = fidelity
    best_rho = rho

    # gradient descent
    grad_angles= best_angles
    n = 0
    index_since_improvement = 0
    while n < N and best_fidelity < eps:
        if verbose: 
            print('n', n)
            print(fidelity, best_fidelity)

        if index_since_improvement % (frac*N)==0: # periodic random search (hop)
            x0 = get_random_angles()
        else:
            gradient = approx_fprime(grad_angles, loss_fidelity, epsilon=1e-8) # epsilon is step size in finite difference
            # if verbose: print(gradient)
            # update angles
            x0 = [best_angles[i] - zeta*gradient[i] for i in range(len(best_angles))]
            grad_angles = x0

        # minimize angles
        best_angles, fidelity, rho = minimize_angles(x0)
        if fidelity > best_fidelity: # if new best, update
            best_fidelity = fidelity
            best_angles = best_angles
            best_rho = rho
            index_since_improvement = 0
        else: # if not new best, increment index
            index_since_improvement += 1

        n+=1 # increment iteration
    if verbose:
        print('Best fidelity: ', best_fidelity)
        print('Best angles: ', best_angles)
        print('Best rho: ', best_rho)
        print('Actual rho: ', targ_rho)

    return targ_name, n, best_fidelity, best_angles[0], best_angles[1], best_angles[2], best_angles[3], best_rho, targ_rho

if __name__=='__main__':
    targ = get_E1(np.pi/4, np.pi/2)
    targ_name = 'E1_60_90'
    alec_decompose(targ, targ_name, verbose=True)

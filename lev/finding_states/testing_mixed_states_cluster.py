# Imports
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import random
import sympy as sp
from multiprocessing import cpu_count, Pool

from uncertainties import ufloat
from uncertainties import unumpy as unp

# Original code adapted from Oscar Scholin in Summer 2023
from sample_rho import *
from rho_methods import *

### Helper Functions ###
def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)


def get_theo_rho(state, eta, chi):
    '''
    Calculates the density matrix (rho) for a given set of paramters (eta, chi) for Stuart's states
    
    Parameters:
    state (string): Which state we want
    eta (float): The parameter eta.
    chi (float): The parameter chi.
    
    Returns:
    numpy.ndarray: The density matrix (rho)
    '''
    # Define kets and bell states in vector form 
    H = ket([1,0])
    V = ket([0,1])
    R = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
    L = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
    D = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1)])
    A = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1)])
    
    PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)
    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)
    
    ##  The following 2 states inspired the Ws
    
    if state == 'phi plus, phi minus':
        phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PHI_MINUS
    
    if state == 'psi plus, psi minus':
        phi = np.cos(eta)*PSI_PLUS + np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    ## The following 6 states inspired the W primes
    
    if state == 'phi plus, psi minus':
        phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    if state == 'phi minus, psi plus':
        phi = np.cos(eta)*PHI_MINUS + np.exp(1j*chi)*np.sin(eta)*PSI_PLUS
    
    if state == 'phi plus, i psi plus':
        phi = np.cos(eta)*PHI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_PLUS
    
    if state == 'phi plus, i phi minus':
        phi = np.cos(eta)*PHI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*PHI_MINUS

    if state == 'psi plus, i psi minus':
        phi = np.cos(eta)*PSI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    if state == 'phi minus, i psi minus':
        phi = np.cos(eta)*PHI_MINUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    ## The following state(s) are an attempt to find new positive W negative W prime states.
    if state == 'HR_VL':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,R) + (1 - np.exp(1j*chi))/2 * np.kron(V,L)
    
    if state == 'HR_iVL':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,R) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,L)
    
    if state == 'HL_VR':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,L) + (1 - np.exp(1j*chi))/2 * np.kron(V,R)
        
    if state == 'HL_iVR':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,L) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,R)
        
    if state == 'HD_VA':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,D) + (1 - np.exp(1j*chi))/2 * np.kron(V,A)
    
    if state == 'HD_iVA':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,D) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,A)
        
    if state == 'HA_VD':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,A) + (1 - np.exp(1j*chi))/2 * np.kron(V,D)
    
    if state == 'HA_iVD':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,A) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,D)
    
    # create rho and return it
    rho = phi @ phi.conj().T
    return rho
    
def generate_state(state_list, state_prob, eta_chi):
    '''
    Uses above helper functions to generate a given mixed state
    
    Parameters:
    state_list (list): list of state names that are to be mixed, must match creatable state names above
    state_prob (list): probability of each state being mixed in state_list, must match index
    eta_chi (list): what eta and chi to use for each state, must match index
    
    Returns:
    rho: an NxN density matrix
    '''
    
    # get individual rho's per state in state_list, taking probability into account
    individual_rhos = []
    for i, state in enumerate(state_list):
        individual_rhos.append(state_prob[i] * get_theo_rho(state, *eta_chi))
    # sum all matrices in individual rhos
    rho = np.sum(individual_rhos, axis = 0)
    
    return rho
        
    
def create_noise(rho, power):
    '''
    Adds noise of order power to a density matrix rho
    
    Parameters:
    rho: NxN density matrix
    power: integer multiple of 10
    
    Returns:
    noisy_rho: rho with noise
    '''
    
    # get size of matrix
    n, _ = rho.shape
    
    # iterature over matrix and add some random noise to each elemnent
    for i in range(n):
        for j in range(n):
            rando = random.random() / (10 ** power)
            rho[i,j] += rando
    noisy_rho = rho
    
    return noisy_rho

######## Helper methods for analyzing density matrices
def analyze_rho(rho_actual, verbose = False, id='id'):
    '''; 
    __
    inputs:
        filenames: list of filenames to analyze
        settings: dict of settings for the experiment
        id: str, special identifier of experiment; used for naming the df
    __
    returns: df with:
        - W theory (adjusted for purity) and W expt and W unc
        - W' theory (adjusted for purity) and W' expt and W' unc
    '''
    
    # calculate W and W' theory
    W_T_ls, W_params = compute_witnesses(rho = rho_actual, return_all = True, return_params = True) # theory #, return_all = True, return_params = True
    
    # parse lists
    W_min = min(W_T_ls[:6])
    Wp_t1 = min(W_T_ls[6:9])
    Wp_t2 = min(W_T_ls[9:12])
    Wp_t3 = min(W_T_ls[12:15])
    
    if verbose:
        # If verbose, return not just the 4 values but also the names of the minimal witnesses!
        # Define dictionary to get name of
        all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9']
        index_names = {i: name for i, name in enumerate(all_W)}

        # Get which W/W' were minimized
        W_min_name = index_names.get(W_T_ls.index(W_min), 'Unknown')
        Wp1_min_name = index_names.get(W_T_ls.index(Wp_t1), 'Unknown')
        Wp2_min_name = index_names.get(W_T_ls.index(Wp_t2), 'Unknown')
        Wp3_min_name = index_names.get(W_T_ls.index(Wp_t3), 'Unknown')
        
        # Return the params as well, add in if return_params = True in compute_witness call
        W_param = W_params[0]
        Wp1_param = W_params[1]
        Wp2_param = W_params[2]
        Wp3_param = W_params[3]
        
        # Find names from dictionary and return them and their values
        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp1_param, Wp2_param, Wp3_param
    else:
         return W_min, Wp_t1, Wp_t2, Wp_t3

def find_states(states, prob, state_name):
    '''
    Find states based on given parameters, adapted from any_state_gen.ipynb for multiprocessing

    Parameters:
    - states (list): List of states.
    - prob (float): Probability value.
    - state_name (str): Name of the state in terms of eta and chi.

    Returns:
    - list: List containing information about the found state.

    '''
    # get the state's density matrix
    #rad_angles = state_name[k]
   
    rho_actual = generate_state(states, prob, state_name)
    
    # get the important info from the state
    W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp1_param, Wp2_param, Wp3_param = analyze_rho(rho_actual, verbose = True)
    
    if W_min > 0.01:
        if Wp_t1 < -0.01 or Wp_t2 < -0.01 or Wp_t3 < -0.01:
            values = [Wp_t1, Wp_t2, Wp_t3]
            params_prime = [Wp1_param, Wp2_param, Wp3_param]
            names_list = ['Wp1', 'Wp2', 'Wp3']
            min_name, min_value, min_param = min(zip(names_list, values, params_prime), key = lambda pair: pair[1])
            
            # print out if it was a state we're saving
            print('A pretty good state was:',  [states, state_name, prob, [W_min_name, W_min, W_param], [min_name, min_value, min_param]])
            
            return [states, state_name, prob, [W_min_name, W_min, W_param], [min_name, min_value, min_param]]
    else:
        return None
if __name__ == '__main__':
    #print(multiprocessing.cpu_count())
    #  Instantiate all the things we need
    list_of_creatable_states = ['phi plus, phi minus', 'psi plus, psi minus', 'HR_VL', 'HR_iVL', 'HL_VR', 'HL_iVR', 'HD_VA', 'HD_iVA', 'HA_VD', 'HA_iVD'] #
    
    etas = [np.pi/4] #np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2
    chis = np.linspace(0.001, np.pi/2, 6)
    num_etas = len(etas)
    num_chis = len(chis)

    probs = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]

    # Instantiate states to sweep over for every mixed state
    states_names = []
    states = []
    for i, eta in enumerate(etas): 
        for chi in chis:
            states_names.append([np.rad2deg(eta), np.rad2deg(chi)])
            states.append([eta, chi]) 
            
    #  Create a full list of inputs, structured each as [[state_1,state_2], [eta,chi], [prob_1,prob_2]].
    inputs = []
    list_of_mixed_states = []
    for i, state_1 in enumerate(list_of_creatable_states):
        for j, state_2 in enumerate(list_of_creatable_states):
            for k, prob in enumerate(probs):
                for l, state_name in enumerate(states_names):
                    inputs.append([[state_1, state_2],state_name, prob])
    
    pool = Pool(cpu_count())
    results = pool.starmap_async(find_states, inputs).get()

    ## end multiprocessing ##
    pool.close()
    pool.join()
    print(results)
    # filter None results out
    results = [result for result in results if result is not None] 

    # build df
    columns = ['states', 'eta-chi', 'probabilities', 'W', 'Wp']
    df = pd.DataFrame.from_records(results, columns = columns)
    print('saving!')


    df.to_csv(f'paper_states/creatable_state_run/all_good_states.csv', index=False)
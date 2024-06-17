# Imports
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import sympy as sp

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

def plot_all(name, etas, chis = [], eta_sweep=False):
    '''
    Plots data from main
    
    Parameters:
    names (list): which csv names to read
    etas (list): list of eta values
    chis (list): list of chi values, only used if eta_sweep=True
    eta_sweep (bool): whether we want to produce a plot of witness val versus eta
    '''
    fig, ax = plt.subplots()
    
    
    # differentiate between a sweep over eta or chi
    if eta_sweep == True:
        chis_nice = []
        for i, chi in enumerate(chis):
            # Read in CSV
            data = pd.read_csv(f'paper_states/{name}_{chis[i]}.csv')
            # Extract data
            W = data['W']
            min_Wp = data['min_W_prime']
            eta = data['eta_arr']
            chis_nice.append(round(np.degrees(chi)))
            # Plot data
            ax.plot(eta, W, label = f'$W, \chi = {chis_nice[i]}$')
            ax.plot(eta, min_Wp, label = f'$W\prime, \chi = {chis_nice[i]}$')
    else:   
        etas_nice = []     
        for i, eta in enumerate(etas):
            
            # Read in CSV
            data = pd.read_csv(f'paper_states/{name}_{etas[i]}.csv')
            # Extract data
            W = data['W']
            min_Wp = data['min_W_prime']
            chi = data['chi']

            # Plot data
            etas_nice.append(round(np.degrees(eta)))
            ax.plot(chi, W, label = f'$W, \eta = {etas_nice[i]}$')
            ax.plot(chi, min_Wp, label = f'$W\prime, \eta = {etas_nice[i]}$')
    if not eta_sweep:
        name = f'{name}_{etas_nice[0]}'  #_{etas_nice[1]}_{etas_nice[2]}_{etas_nice[3]}
    if eta_sweep:
        name = f'{name}_{chis_nice}'
    print(name)
    ax.axhline(0, color='black', linewidth=0.5) 
    ax.set_title(f'State {name}', fontsize=12)
    ax.set_ylabel('Witness value', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    ax.legend(fontsize=8, loc = 'upper right')
    ax.set_xlabel('$\chi$', fontsize=12)
    if eta_sweep:
        ax.set_xlabel('$\eta$',)
    plt.savefig(f'paper_states/{name}.pdf')

if __name__ == '__main__':
    #  Instantiate all the things we need
    list_of_creatable_states = ['phi plus, phi minus', 'psi plus, psi minus', 'HR_VL', 'HR_iVL', 'HL_VR', 'HL_iVR', 'HD_VA', 'HD_iVA', 'HA_VD', 'HA_iVD']

    etas = [np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    chis = np.linspace(0.001, np.pi/2, 6)
    num_etas = len(etas)
    num_chis = len(chis)

    probs = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]

    # Instantiate states to sweep over for every mixed state
    states_names = []
    states = []
    for i, eta in enumerate(etas): 
        for chi in chis:
            states_names.append((np.rad2deg(eta), np.rad2deg(chi)))
            states.append((eta, chi)) 

    # add the state to this list whenever W is positive and W prime is negative
    # each addition is of the form [[state_name_1, state_name_2], [chi, eta], [prob_1, prob_2], [W_witness_name, W_witness_value, theta], 
    #                               [Wp_witness_name, Wp_witness_value, theta]]
    Wh_Wpl = []
    special_list = [] # this is if W > 0.2 and Wp < 0.2 (make it more obvious which are good states)
    for i, state_1 in enumerate(list_of_creatable_states):
        for j, state_2 in enumerate(list_of_creatable_states):
            for l, prob in enumerate(probs):
                for k, state_set in enumerate(states_names):
                    arr_to_add = []
                    # get the state's density matrix
                    rad_angles = states[k]
                    names = [state_1, state_2]
                    rho_actual = generate_state(names, prob, rad_angles)
                    print('i am werking')
                    # get the important info from the state
                    W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp1_param, Wp2_param, Wp3_param = analyze_rho(rho_actual, verbose = True)
                    
                    if W_min > 0.01:
                        if Wp_t1 < -0.01 or Wp_t2 < -0.01 or Wp_t3 < -0.01:
                            values = [Wp_t1, Wp_t2, Wp_t3]
                            params_prime = [Wp1_param, Wp2_param, Wp3_param]
                            names_list = ['Wp1', 'Wp2', 'Wp3']
                            min_name, min_value, min_param = min(zip(names_list, values, params_prime), key = lambda pair: pair[1])
                            append_this = [[names[0], names[1]], [state_set], prob, [W_min_name, W_min, W_param], [min_name, min_value, min_param]]

                            Wh_Wpl.append(append_this)
                            print('A pretty good state was:', append_this)
                    # if super good attach to special list!
                    if W_min > 0.2:
                        if Wp_t1 < -0.2 or Wp_t2 < -0.2 or Wp_t3 < -0.2:
                            values = [Wp_t1, Wp_t2, Wp_t3]
                            params_prime = [Wp1_param, Wp2_param, Wp3_param]
                            names_list = ['Wp1_min_name', 'Wp2_min_name', 'Wp3_min_name']
                            min_name, min_value, min_param = min(zip(names_list, values, params_prime), key = lambda pair: pair[1])
                            append_this = [[names[0], names[1]], [state_set], prob, [W_min_name, W_min, W_param], [min_name, min_value, min_param]]
                            
                            special_list.append(append_this)
                            print('An amazing state was:', append_this)
                    
    # save it all to csvs!                       
    df_to_save = pd.DataFrame(Wh_Wpl)
    df_to_save_special = pd.DataFrame(special_list)

    df_to_save.to_csv(f'paper_states/creatable_state_run/all_states_new.csv', index=False)
    df_to_save_special.to_csv(f'paper_states/creatable_state_run/special_states_new.csv', index = False)
'''
AuthorL Lev Gruber
Last Update: 5/17/2024

This file uses code from Oscar's process_expt_richard.py in Summer 2023 to brute force
calculate eta and chi values (for the state cos(eta)PHI+ + sin(eta)e^i*chi PHI-) in which W is positive
and W' is negative. 

It does so by:
1. Creating theoretical density matrices at a variety of eta values and 1-2 chi values
2. Computing both W and W' witnesses for each density matrix
3. Pushing values with +W and -W' into one df, and values with +W, +W' into another. 

In the future, I hope to add functionality to compute V witnesses for states for which W and W' are positive.
'''

# Imports
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

from uncertainties import ufloat
from uncertainties import unumpy as unp

# Original code adapted from Oscar Scholin in Summer 2023
from sample_rho import *
from rho_methods import *

####### Helper methods for rho creation
def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho_psi(eta, chi):
    """
    Calculates the density matrix (rho) for a given set of parameters (eta, chi) in Psi.

    Parameters:
    eta (float): The parameter eta.
    chi (float): The parameter chi.

    Returns:
    numpy.ndarray: The density matrix (rho) calculated based on the given parameters.
    """
    H = ket([1,0])
    V = ket([0,1])
    
    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)

    psi = np.cos(eta)*PSI_PLUS + np.exp(1j*chi)*np.sin(eta)*PSI_MINUS

    rho = psi @ psi.conj().T

    return rho

def get_theo_rho_phi(eta, chi):
    """
    Calculates the density matrix (rho) for a given set of parameters (eta, chi) in Phi.

    Parameters:
    eta (float): The parameter eta.
    chi (float): The parameter chi.

    Returns:
    numpy.ndarray: The density matrix (rho) calculated based on the given parameters.
    """
    # HV Kets
    H = ket([1,0])
    V = ket([0,1])
    
    # Bell States
    PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)
    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)
    
    phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PHI_MINUS

    rho = phi @ phi.conj().T

    return rho

def get_theo_rho_stu(state, eta, chi):
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
    
    PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)
    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)
    
    # create the state PHI+ + PSI-
    if state == 'phi plus, psi minus':
        phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
        rho = phi @ phi.conj().T
        return rho
    
    # create the state PHI- + PSI+
    if state == 'phi minus, psi plus ':
        phi = np.cos(eta)*PHI_MINUS + np.exp(1j*chi)*np.sin(eta)*PSI_PLUS
        rho = phi @ phi.conj().T
        return rho
    
    # create the state PSI+ + iPSI-
    if state == 'psi plus, i psi minus':
        phi = np.cos(eta)*PSI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
        rho = phi @ phi.conj().T
    
    if state == 'phi minus, i psi minus':
        phi = np.cos(eta)*PHI_MINUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
        rho = phi @ phi.conj().T
        
    else:
        return 'gimme a state'
    
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
    W_T_ls, W_params = compute_witnesses(rho = rho_actual, return_all = True, return_params = True) # theory
    #print(W_T_ls)
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
        
        # Return the params as well
        W_param = W_params[0]
        Wp1_param = W_params[1]
        Wp2_param = W_params[2]
        Wp3_param = W_params[3]
        
        # Find names from dictionary and return them and their values
        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp1_param, Wp2_param, Wp3_param
    else:
         return W_min, Wp_t1, Wp_t2, Wp_t3

def plot_all(W, min_Wp, eta, chi)
    '''
    Plots data from main
    
    Parameters:
    W (array): 
    min_Wp (array):
    eta (array):
    chi ()
    '''

####### Main Functions

if __name__ == '__main__':
    # Determine all sweeping settings and choose name for file
    etas = [np.pi/6]
    chis = np.linspace(0.001, np.pi/2, 6)
    name = f'phi_plus_psi_minus_{etas[0]}'
    plot = True
    
    # Instantiate states to sweep over
    states_names = []
    states = []
    for eta in etas: 
        for chi in chis:
            states_names.append((np.rad2deg(eta), np.rad2deg(chi)))
            states.append((eta, chi)) 
    
    # Obtain the density matrix for each state; current making states of form psi
    rho_actuals = []
    for i, state_n in enumerate(states_names):
        rad_angles = states[i]
        rho_actuals.append(get_theo_rho_stu('phi plus, psi minus',rad_angles[0],rad_angles[1]))

    # Instantiate lists to save as csv
    wpl = [] 
    eta_arr = []
    chi_arr = []
    wm_arr = []
    min_W = []
    min_Wp1 = []
    min_Wp2 = []
    min_Wp3 = []
    
    # Create noisy rhos
    #for i, rho in enumerate(rho_actuals):
        #rho_actuals[i] = create_noise(rho, 3)
        #if i == 0:
            #print(rho_actuals[i])
    
    # Save witness values to above lists
    for i, rho in enumerate(rho_actuals):
        # find the minimum witness expectation value of the 3 w primes and of the W.
        WM, WP1, WP2, WP3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp1_param, Wp2_param, Wp3_param = analyze_rho(rho, verbose = True)
        e, c = states[i]
        min_wp = min(WP1, WP2, WP3)
                     
        # add to arrays to save
        wpl.append([min_wp])
        eta_arr.append([e])
        wm_arr.append([WM])
        chi_arr.append([c])
        min_W.append([W_min_name])
        min_Wp1.append([Wp1_min_name])
        min_Wp2.append([Wp2_min_name])
        min_Wp3.append([Wp3_min_name])
        print('The current CHI is:', chis[i])
        print('Minimum W was:', W_min_name, 'with value', WM, 'and theta', W_param)
        #print('Minimum W prime 1 was:', Wp1_min_name, 'with value', WP1, 'and theta', Wp1_param)
        #print('Minimum W prime 2 was:', Wp2_min_name, 'with value', WP2, 'and theta', Wp2_param)
        #print('Minimum W prime 3 was:', Wp3_min_name, 'with value', WP3, 'and theta', Wp3_param)
        
    # Save data into a csv
    data = {'W': wm_arr, 'min_W_prime': wpl, 'eta': eta_arr, 'chi': chi_arr}
    df = pd.DataFrame(data)
    df.to_csv(f'stu_states/{name}.csv', index=False)
    
    if plot == True:
        plot_all()
    # Save to CSV
    
    




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

from uncertainties import ufloat
from uncertainties import unumpy as unp

    # Original code from Oscar Scholin in Summer 2023
from sample_rho import *
from rho_methods import *

# Helper methods 
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
    H = ket([1,0])
    V = ket([0,1])
    
    PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)

    phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PHI_MINUS

    rho = phi @ phi.conj().T

    return rho


def analyze_rho(rho_actual, id='id'):
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
    W_T_ls = compute_witnesses(rho = rho_actual) # theory

    # parse lists
    W_min_T = W_T_ls[0]
    Wp_t1_T = W_T_ls[1]
    Wp_t2_T = W_T_ls[2]
    Wp_t3_T = W_T_ls[3]
    
    return W_min_T, Wp_t1_T, Wp_t2_T, Wp_t3_T

if __name__ == '__main__':
    # Sweeping parameters, desied plot x-axis being eta
    #etas = np.linspace(0.001, np.pi/2, 90)
    #chis = [np.pi/4]
    
    # Sweeping parameters, desired plot x-axis being chi
    etas = [np.pi/4]
    chis = np.linspace(0.001, np.pi/2, 90)
    
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
        rho_actuals.append(get_theo_rho_psi(rad_angles[0],rad_angles[1]))

    # Instantiate lists to save as csv
    wpl = [] 
    eta_arr = []
    chi_arr = []
    wm_arr = []
    
    # Save witness values to above lists
    for i, rho in enumerate(rho_actuals):
        # find the minimum witness expectation value of the 3 w primes and of the W.
        WM, WP1, WP2, WP3 = analyze_rho(rho)
        e, c = states[i]
        min_wp = min(WP1, WP2, WP3)
        
        # add to arrays to save
        wpl.append([min_wp])
        eta_arr.append([e])
        wm_arr.append([WM])
        chi_arr.append([c])
        
    # Save data into a csv
    data = {'W': wm_arr, 'min_W_prime': wpl, 'eta': eta_arr, 'chi': chi_arr}
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('psi_chisweep_45eta.csv', index=False)
    




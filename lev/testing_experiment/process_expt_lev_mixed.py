# file to read and process experimentally collected density matrices specifically for mixed states
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

from uncertainties import ufloat
from uncertainties import unumpy as unp

from sample_rho import *
from rho_methods import *
from process_expt_lev import *

current_path = dirname(abspath(__file__))
DATA_PATH = 'rho_5202024_INSERT'

def simple_analyze_rho(rho_actual, verbose = False, id='id'):
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
    W_T_ls = compute_witnesses(rho = rho_actual, return_all = True) # theory #, return_all = True, return_params = True
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

def plot_all(name, etas):
    '''
    Plots data from main
    
    Parameters:
    names (list): which csv names to read
    
    W (array): Minimum W array
    min_Wp (array): Minimum W prime array
    eta (float): eta value 
    chi (array): chi array
    name (string): name for file output (of state)
    '''
    fig, ax = plt.subplots()
    etas_nice = []
    for i, eta in enumerate(etas):
        # Read in CSV
        data = pd.read_csv(f'mixed_states/{name}_{etas[i]}.csv')
        # Extract data
        W = data['W']
        min_Wp = data['min_W_prime']
        chi = data['chi']

        # Plot data
        etas_nice.append(round(np.degrees(eta)))
        ax.plot(chi, W, label = f'$W, \eta = {etas_nice[i]}$')
        ax.plot(chi, min_Wp, label = f'$W\prime, \eta = {etas_nice[i]}$')
  
    name = f'{name}_{etas_nice[0]}'  #_{etas_nice[1]}_{etas_nice[2]}_{etas_nice[3]}
    print(name)
    ax.axhline(0, color='black', linewidth=0.5) 
    ax.set_title(f'State {name}', fontsize=12)
    ax.set_ylabel('Witness value', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    ax.legend(fontsize=8, loc = 'upper right')
    ax.set_xlabel('$\chi$', fontsize=12)
    plt.savefig(f'pure_and_mixed_states/{name}.pdf')


if __name__ == '__main__':
    # define expeirmental set up parameters
    etas = [np.pi/4]
    chis = np.linspace(0.001, np.pi/2, 6)
    print(chis)
    states_names = []
    states = []
    num_states = 2 # num states must be the number of pure states being mixed and match len(probs)
    probs = [0.65, 0.35]
    file_name = 'mixed_0.65phi_0.35psi'


    # gather all states created according to alpha and beta (chi and eta)
    for eta in etas:
        for chi in chis:
            states_names.append((np.rad2deg(eta), np.rad2deg(chi)))
            states.append((eta, chi))

    # this is specifically for two pure state mixed states, can generalize later
    # obtain file names list for later use, and get theoretical rho matrices
    filenames = [[], []]
    settings = []
    rho_actuals = [[], []]
    for i, state_n in enumerate(states_names):
        filenames[0].append(f"rho_('E0', {state_n})_psi.npy")
        settings.append([state_n[0],state_n[1]])
        rad_angles = states[i]
        rho_actuals[0].append(get_theo_rho(rad_angles[0],rad_angles[1]))
        
    # do the same for our second state
    for i, state_n in enumerate(states_names):
        filenames[1].append(f"rho_('E0', {state_n})_phi.npy")
        settings.append([state_n[0],state_n[1]])
        rad_angles = states[i]
        rho_actuals[1].append(get_theo_rho(rad_angles[0],rad_angles[1]))
        
    # get rhos 
    rho_expt = [[], []]
    for i in range(2):
        for file in filenames[i]:
            trial, rho, unc, Su, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            rho_expt[i].append(probs[i]*rho) # use the above defined probabilities
    # pairwise add each rho from the two lists, obtaining
    mixed_rho = [sum(x) for x in zip(*rho_expt)]
    
    #### Must add functionality for propagating means of purities, fidelities, and other ####
    
     # Instantiate lists to save as csv
    W_arr = []  # lowest W value (4 is number of etas)
    Wp_arr = []     # lowest W prime value
    eta_arr = []
    chi_arr = []

    ####### Witness each rho and save it
    
    ### Save witness values to above lists of lists
    for j, rho in enumerate(mixed_rho):
        # find the minimum witness expectation value of the 3 w primes and of the W.
        # Add in to get which W was minimum: W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp1_param, Wp2_param, Wp3_param,  verbose = True
        e = etas[i]
        c = chis[j]
        WM, WP1, WP2, WP3 = simple_analyze_rho(rho) 
        min_wp = min(WP1, WP2, WP3)
        print('My Ws were:', WM, min_wp, 'at eta and chi:', e, c)
        # Assuming that the index i corresponds to the eta trial number (0 to 3)
        W_arr[i].append(WM)
        Wp_arr[i].append(min_wp)
        eta_arr[i].append(e)
        chi_arr[i].append(c)

    # Save each list to a separate CSV file
    
    data = pd.DataFrame({
        'W': W_arr[i],
        'min_W_prime': Wp_arr[i],
        'eta_arr': eta_arr[i],
        'chi': chi_arr[i]
    })
    data.to_csv(f'mixed_states/{file_name}_{etas[0]}.csv', index=False)
    plot_all(file_name, etas)
    
    # analyze rho files
    id = 'rho_5292024'
    for rho in mixed_rho:
        simple_analyze_rho(rho, id=id)
    make_plots_E0(f'fixed_rho_analysis_{id}.csv')
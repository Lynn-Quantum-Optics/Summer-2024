# file to mix pure state rhos into mixed states and save them as csvs
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
DATA_PATH = 'hrivl7-19_havd_mix'
    
def mix_states(file_names, probs, state_name):
    '''
    This returns mixed rhos and saves a csv with a mixed state.
    Note that uncertainty is taken as a mean, as is purity and fidelity.
    This file assumes one chi is being analyzed at a time.
    
    Parameters:
    file_names (list of lists): For N files, N lists of lists of the file names of each file,
                                organized in order of chi.
    probs (list): probability of the rho in each file.
    state_name (str): name of state to be used when saving data
    
    Returns:
    rhos (list): list of merged rhos
    '''
    
    # Obtain the data from each file, and use probs: 
    row_names = ['rho', 'unc', 'Su', 'un_proj', 'un_proj_unc', '_', 'angles', 'fidelity', 'purity']
    df = pd.DataFrame(index=row_names, columns=file_names) 
    for i, file in enumerate(file_names):
        data = np.load(join(DATA_PATH, file), allow_pickle=True)
        for j, row_name in enumerate(row_names):
            df.loc[row_name, file] = data[j]
        # Use probabilities on all values that I felt like
        df.loc['rho', file] = probs[i] * df.loc['rho', file]
        df.loc['unc', file] = probs[i] * df.loc['unc', file]
        df.loc['un_proj', file] = probs[i] * df.loc['un_proj', file]
        df.loc['un_proj_unc', file] = probs[i] * df.loc['un_proj_unc', file]
        df.loc['purity', file] = probs[i] * df.loc['purity', file]
        df.loc['fidelity', file] = probs[i] * df.loc['fidelity', file]
    # Create a new data frame and propagate it with previous data frames values
    df_to_save = pd.DataFrame(index = row_names)
    df_to_save['col'] = df[file_names[0]]
    for i, file_name in enumerate(file_names):
        if file_name == file_names[0]:  # Skip the first bc that is already in the new df
            pass
        else:
            df_to_save['col'] += df[file_name]
    
 
    # normalize the angle values

    rho, unc, Su, un_proj, un_proj_unc, state, angles, fidelity, purity = df_to_save.iloc[:, 0] 
    print(rho, unc, Su, un_proj, un_proj_unc, state, angles, fidelity, purity)
 
    np.save(join(DATA_PATH,f"rho_('E0', {state_name})_3"), df_to_save['col'].to_numpy())
    #save results
    #with open(f"int_state_sweep_phi45/rho_('E0', {state_n})_1.npy", 'wb') as f:
        #np.save(f, (rho, unc, Su, un_proj, un_proj_unc, state, angles, fidelity, purity))
    print(df.head(10))
    print(df_to_save.head(10))
    
if __name__ == '__main__':
    # define experimental parameters
    etas = [np.pi/4]
    chis = np.linspace(0.001, np.pi/2, 6)
    probs = [0.65, 0.35]
    states_names = []
    states = []

    # define state names
    for eta in etas:
        for chi in chis:
            states_names.append((np.rad2deg(eta), np.rad2deg(chi)))
            states.append((eta, chi))
    # get new csvs of mixed state
    for i, state_n in enumerate(states_names):
        state_n = (float(state_n[0]), float(state_n[1]))

        filenames = []
        filenames.append(f"rho_('E0', {state_n})_1.npy")
        filenames.append(f"rho_('E0', {state_n})_2.npy") 
        rad_angles = states[i]
        mix_states(filenames, probs, state_n)
    







    
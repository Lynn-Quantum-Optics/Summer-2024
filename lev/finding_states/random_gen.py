"""
This file uses the ToQITo toolkit to generate random pure states, then witnessing them with W and W'.
"""
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import sympy as sp
from toqito.rand import random_density_matrix, random_state_vector
from toqito.matrix_props import is_density
from toqito.state_props import is_pure

from uncertainties import ufloat
from uncertainties import unumpy as unp

# Original code adapted from Oscar Scholin in Summer 2023
from sample_rho import *
from rho_methods import *

def gen_rhos(num, dim = 4):
    '''
    Generates (num) number of density matrices using toqito
    
    Parameters:
    num (int): Number of rhos
    
    Returns:
    rhos (list): List of rhos
    '''
    rhos = []
    for _ in range(num):
        rho = random_density_matrix(dim)
        rhos.append(rho)
    return rhos

def gen_pure_rhos(num, dim = 4):
    '''
    Generates (num) number of density matrices of pure states using toqito
    
    Parameters:
    num (int): Number of rhos
    
    Returns:
    rhos (list): List of rhos
    '''
    rhos = []
    for _ in range(num):
        vec = random_state_vector(dim)
        rho = vec @ vec.conj().T
        rhos.append(rho)
    return rhos

def analyze_rho(rho, verbose = False):
    '''
    Uses compute_witness to find the minimal witness values for a given rho
    
    Parameters:
    rho (array): density matrix
    verbose = True (bool): Whether to return all witness values
    
    Return:
    W_min (float): Minimum W value
    Wp_min (float): Minimum W prime value
    '''
     # calculate W and W'
    W_T_ls = compute_witnesses(rho = rho, return_all = True) # can include return_params for minimization params
    # parse lists
    W_min = min(W_T_ls[:6])
    Wp_t1 = min(W_T_ls[6:9])
    Wp_t2 = min(W_T_ls[9:12])
    Wp_t3 = min(W_T_ls[12:15])
    
    Wp_min = min(Wp_t1, Wp_t2, Wp_t3)
    
    if verbose:
        # Define dictionary to get name of
        all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9']
        index_names = {i: name for i, name in enumerate(all_W)}

        # Get which W/W' were minimized
        W_min_name = index_names.get(W_T_ls.index(W_min), 'Unknown')
        Wp1_min_name = index_names.get(W_T_ls.index(Wp_t1), 'Unknown')
        Wp2_min_name = index_names.get(W_T_ls.index(Wp_t2), 'Unknown')
        Wp3_min_name = index_names.get(W_T_ls.index(Wp_t3), 'Unknown')
        
        # Find names from dictionary and return them and their values
        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name
    else:
         return W_min, Wp_min
     
if __name__ == '__main__':
    #### Choose number of rhos & whether pure
    rho_num = 100
    pure = False
    
    #### Generate rhos
    if pure == True:
        rhos = gen_pure_rhos(rho_num)
    else:
        rhos = gen_rhos(rho_num)
    
    #### Witness states
    num_witnessed_W = 0
    num_witnessed_Wp = 0
    
    for rho in rhos:
        W_min, Wp_min = analyze_rho(rho)
        if W_min < 0:
            num_witnessed_W += 1
        if Wp_min < 0:
            num_witnessed_Wp += 1
    
    #### Calculate proportions
    percent_W = num_witnessed_W / rho_num * 100
    percent_Wp = num_witnessed_Wp / rho_num * 100
    
    print('Ws witnessed', percent_W, '% of total states, or', num_witnessed_W, 'out of', rho_num)
    print('W primes witnessed', percent_Wp, '% of total states, or', num_witnessed_Wp, 'out of', rho_num)
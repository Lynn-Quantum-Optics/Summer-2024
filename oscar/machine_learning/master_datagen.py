## file to standardize random generation of density matrices. replaces jones_simplex_datagen.py and roik_datagen.py ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import join, isdir

from multiprocessing import cpu_count, Pool

from rho_methods import *
from random_gen import *
from jones import *

def build_dataset(random_method, prob_type, num_to_gen, savename):
    ''' Fuction to build a dataset of randomly generated states.
    params:
        random_method: string, either 'simplex', 'jones_I','jones_C' or 'random'
        prob_type: string, either 'standard' for nonredudant 9, or 'roik_like' for 16 nonredudant
        num_to_gen: number of states to generate
        savename: name of file to save data to
    '''

    # confirm valid random method
    assert random_method in ['simplex', 'jones_I','jones_C', 'random'], f'Invalid random method. You have {random_method}.'

    # confirm valid prob_type
    assert prob_type in ['standard', 'roik_like'], f'Invalid prob_type. You have {prob_type}.'

    ## initialize dataframe to hold states ##
    # because of my Hoppy model in jones.py, we no longer need to save the generating angles; we can determine them!! :)) This doesn't work with mixed states, but we can still get a closest aproximation.
    if prob_type == 'standard':
        df_cols = ['HH', 'HV', 'VV', 'DD', 'DA', 'AA', 'RR', 'RL', 'LL', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity']
    elif prob_type == 'roik_like':
        df_cols = ['HH', 'VV', 'HV', 'DD', 'AA', 'RR', 'LL', 'DL', 'AR', 'DH', 'AV', 'LH', 'RV', 'DR', 'DV', 'LV' ,'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity']
        
    ## generate states ##
    if random_method == 'simplex':
        func = get_random_simplex 
    elif random_method=='jones_C':
        func = get_random_jones(setup='C')
    elif random_method=='jones_I':
        func = get_random_jones(setup='I')
    elif random_method=='roik':
        func = get_random_roik
    elif random_method=='werner_simplex':
        func = get_random_werner_simplex

    def gen_rand_info():
        ''' Function to ompute random state based on imput method and return measurement projections, witness values, concurrence, and purity.'''

        # generate random state
        rho = func()
        W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho)
        concurrence = get_concurrence(rho)
        purity = get_purity(rho)
        if prob_type=='standard':
            HH, HV, VV, DD, DA, AA, RR, RL, LL = get_9s_projections(rho)
            return HH, HV, VV, DD, DA, AA, RR, RL, LL, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity
        elif prob_type=='roik_like':
            HH, VV, HV, DD, AA, RR, LL, DL, AR, DH, AV, LH, RV, DR, DV, LV = get_16s_projections(rho)
            return HH, VV, HV, DD, AA, RR, LL, DL, AR, DH, AV, LH, RV, DR, DV, LV, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity

    ## build multiprocessing pool ##
    pool = Pool(cpu_count())
    inputs = [None]*num_to_gen
    results = pool.starmap_async(gen_rand_info, inputs)

    ## end multiprocessing ##
    pool.close()
    pool.join()

    # filter None results out
    results = [result for result in results if result is not None] 

    # build df
    df = pd.DataFrame.from_records(results, columns = df_cols)
    df.to_csv(savename+'.csv', index=False)

## ask user for info ##
if __name__=='__main__':
    # random_method, prob_type, num_to_gen, savename
    random_method = input("Enter random method: 'simplex', 'jones_I','jones_C', or 'random': ")
    prob_type = input("Enter prob_type: 'standard' or 'roik_like': ")
    num_to_gen = int(input("Enter number of states to generate: "))
    special = input("Enter special name for file: ")
    if not(isdir('random_gen')):
        os.mkdir('random_gen')
    datadir = bool(int(input('Put in data dir (1) or test dir (0): ')))
    if datadir:
        savename = join('random_gen', 'data', f'{random_method}_{prob_type}_{num_to_gen}_{special}')
    else:
        savename = join('random_gen', 'test', f'{random_method}_{prob_type}_{num_to_gen}_{special}')

    print(f'{random_method}, {prob_type}, {num_to_gen}, {special}')

    build_dataset(random_method, prob_type, num_to_gen, savename)
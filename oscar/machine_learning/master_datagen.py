## file to standardize random generation of density matrices. replaces jones_simplex_datagen.py and roik_datagen.py ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import join, isdir

from multiprocessing import cpu_count, Pool
from functools import partial

from rho_methods import *
from roik_gen import * # actual roik code
from random_gen import *
from jones import *

def gen_rand_info(func, return_prob, do_stokes=False, verbose=True):
    ''' Function to compute random state based on imput method and return measurement projections, witness values, concurrence, and purity.
        params:
            func: function to generate random state
            return_prob: bool, whether to return all 36 probabilities or 15 stokes's parameters
            do_stokes: bool, whether to use stokes's params to calc witnesses or operator
            verbose: bool, whether to print progress
    '''

    # generate random state
    rho = func()
    W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho, do_stokes=do_stokes)
    concurrence = get_concurrence(rho)
    purity = get_purity(rho)
    if verbose: print(f'made state with concurrence {concurrence} and purity {purity}')

    if not(return_prob): # if we want to return stokes's parameters
        II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ = np.real(get_expec_vals(rho).reshape(16,))
        return IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity
    else:
        all_projs = get_all_projs(rho)
        HH = all_projs[0,0]
        HV = all_projs[0,1]
        HD = all_projs[0,2]
        HA = all_projs[0,3]
        HR = all_projs[0,4]
        HL = all_projs[0,5]
        VH = all_projs[1,0]
        VV = all_projs[1,1]
        VD = all_projs[1,2]
        VA = all_projs[1,3]
        VR = all_projs[1,4]
        VL = all_projs[1,5]
        DH = all_projs[2,0]
        DV = all_projs[2,1]
        DD = all_projs[2,2]
        DA = all_projs[2,3]
        DR = all_projs[2,4]
        DL = all_projs[2,5]
        AH = all_projs[3,0]
        AV = all_projs[3,1]
        AD = all_projs[3,2]
        AA = all_projs[3,3]
        AR = all_projs[3,4]
        AL = all_projs[3,5]
        RH = all_projs[4,0]
        RV = all_projs[4,1]
        RD = all_projs[4,2]
        RA = all_projs[4,3]
        RR = all_projs[4,4]
        RL = all_projs[4,5]
        LH = all_projs[5,0]
        LV = all_projs[5,1]
        LD = all_projs[5,2]
        LA = all_projs[5,3]
        LR = all_projs[5,4]
        LL = all_projs[5,5]

        return HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity

def build_dataset(random_method, return_prob, num_to_gen, savename, do_stokes=False, verbose=True):
    ''' Fuction to build a dataset of randomly generated states.
    params:
        random_method: string, either 'simplex', 'jones_I','jones_C' or 'random'
        return_prob: bool, whether to return all 36 probabilities or 15 stokes's parameters
        savename: name of file to save data to
        verbose: bool, whether to print progress
    '''

    # confirm valid random method
    # assert random_method in ['simplex', 'jones_I','jones_C', 'hurwitz'], f'Invalid random method. You have {random_method}.'

    ## initialize dataframe to hold states ##
    # because of my Hoppy model in jones.py, we no longer need to save the generating angles; we can determine them!! :)) This doesn't work with mixed states, but we can still get a closest aproximation.
    if not(return_prob):
        columns = ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity']
    else:
        columns = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity']
        
    ## generate states ##
    if random_method == 'simplex':
        func = get_random_simplex 
    elif random_method=='jones_C':
        func = get_random_jones(setup='C')
    elif random_method=='jones_I':
        func = get_random_jones(setup='I')
    elif random_method=='hurwitz':
        method = int(input('which method for phi random gen do you want? 0, 1, 2: '))
        assert method in [0, 1, 2], f'Invalid method. You have {method}.'
        savename+=f'_method_{method}'
        func = partial(get_random_hurwitz, method=method)
    elif random_method=='roik': # from actual roik code
        func = get_random_rho()

    # build multiprocessing pool ##
    pool = Pool(cpu_count())
    inputs = [(func, return_prob, do_stokes, verbose) for _ in range(num_to_gen)]
    results = pool.starmap_async(gen_rand_info, inputs).get()

    ## end multiprocessing ##
    pool.close()
    pool.join()

    # filter None results out
    results = [result for result in results if result is not None] 

    # build df
    df = pd.DataFrame.from_records(results, columns = columns)
    df.to_csv(savename+'.csv', index=False)

def comp_me_roik():
    ''' Function to compare the states I generate using Roik's method to the ones generated with my method. 
    '''
    num_to_gen = 100000
    run_me = bool(int(input('Run my or Roik code? (1 or 0): ')))
    if run_me:
        savename = '6_27_me'
        rtype='hurwitz'
    else:
        savename = '6_27_roik'
        rtype='roik'

    build_dataset(rtype, True, num_to_gen, savename)



## ask user for info ##
if __name__=='__main__':
    # random_method, return_prob, num_to_gen, savename
    do_roik = bool(int(input('Custom (0) or Compare with actual Roik (1): ')))
    if do_roik:
        comp_me_roik()
    else:
        random_method = input("Enter random method: 'simplex', 'jones_I','jones_C', 'random', or 'hurwitz': ")
        return_prob = bool(int(input("Return probabilities (1) or stokes's parameters (0): ")))
        do_stokes = bool(int(input("Do stokes's parameters (1) or operators (0) to calculate witnesses: ")))
        num_to_gen = int(input("Enter number of states to generate: "))
        special = input("Enter special name for file: ")
        datadir = bool(int(input('Put in data dir (1) or test dir (0): ')))

    if not(isdir('random_gen')):
        os.makedirs('random_gen')
    if not(isdir(join('random_gen', 'data'))):
        os.makedirs(join('random_gen', 'data'))
    if not(isdir(join('random_gen', 'test'))):
        os.makedirs(join('random_gen', 'test'))
    
    if datadir:
        savename = join('random_gen', 'data', f'{random_method}_{return_prob}_{num_to_gen}_{special}')
    else:
        savename = join('random_gen', 'test', f'{random_method}_{return_prob}_{num_to_gen}_{special}')

    print(f'{random_method}, {return_prob}, {num_to_gen}, {special}, {do_stokes}')

    build_dataset(random_method, return_prob, num_to_gen, savename, do_stokes=do_stokes)
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

def gen_rand_info(func, return_prob, do_stokes=False, include_w = True, log_params = False, verbose=False, log_roik_prob=False):
    ''' Function to compute random state based on imput method and return measurement projections, witness values, concurrence, and purity.
        params:
            func: function to generate random state
            return_prob: bool, whether to return all 36 probabilities or 15 stokes's parameters
            do_stokes: bool, whether to use stokes's params to calc witnesses or operator
            include_w: bool, whether to include witness values in return
            log_params: bool, whether to 
            verbose: bool, whether to print progress
    '''

    # generate random state
    if not(log_params):rho = func()
    else: rho, params = func()
    
    W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho, do_stokes=do_stokes)
    concurrence = get_concurrence(rho)
    purity = get_purity(rho)
    min_eig = get_min_eig(rho)
    if verbose: print(f'made state with concurrence {concurrence} and purity {purity}')
    
    if include_w:

        if not(return_prob): # if we want to return stokes's parameters
            II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ = np.real(get_expec_vals(rho).reshape(16,))
            return IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity
        else:
            projs = get_all_projs(rho)
            HH, HV, HD, HA, HR, HL = projs[0]
            VH, VV, VD, VA, VR, VL = projs[1]
            DH, DV, DD, DA, DR, DL = projs[2]
            AH, AV, AD, AA, AR, AL = projs[3]
            RH, RV, RD, RA, RR, RL = projs[4]
            LH, LV, LD, LA, LR, LL = projs[5]
            if not(log_roik_prob):
                if not(log_params): 
                    return HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity, min_eig
        

                else: 
                    return HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity, min_eig, params[0], params[1], params[2], params[3]
            else:
                # get roik probabilities; use their code
                roik_probs = get_all_roik_projs_sc(rho)
                r_HH, r_HV, r_HD, r_HA, r_HR, r_HL = roik_probs[0]
                r_VH, r_VV, r_VD, r_VA, r_VR, r_VL = roik_probs[1]
                r_DH, r_DV, r_DD, r_DA, r_DR, r_DL = roik_probs[2]
                r_AH, r_AV, r_AD, r_AA, r_AR, r_AL = roik_probs[3]
                r_RH, r_RV, r_RD, r_RA, r_RR, r_RL = roik_probs[4]
                r_LH, r_LV, r_LD, r_LA, r_LR, r_LL = roik_probs[5]
                if not(log_params):
                    return HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL, r_HH, r_HV, r_HD, r_HA, r_HR, r_HL, r_VH, r_VV, r_VD, r_VA, r_VR, r_VL, r_DH, r_DV, r_DD, r_DA, r_DR, r_DL, r_AH, r_AV, r_AD, r_AA, r_AR, r_AL, r_RH, r_RV, r_RD, r_RA, r_RR, r_RL, r_LH, r_LV, r_LD, r_LA, r_LR, r_LL, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity, min_eig
                else:
                    return HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL, r_HH, r_HV, r_HD, r_HA, r_HR, r_HL, r_VH, r_VV, r_VD, r_VA, r_VR, r_VL, r_DH, r_DV, r_DD, r_DA, r_DR, r_DL, r_AH, r_AV, r_AD, r_AA, r_AR, r_AL, r_RH, r_RV, r_RD, r_RA, r_RR, r_RL, r_LH, r_LV, r_LD, r_LA, r_LR, r_LL, W_min, Wp_t1, Wp_t2, Wp_t3, concurrence, purity, min_eig, params[0], params[1], params[2], params[3]
    else:
        if not(log_params): return concurrence, purity, min_eig
        
        else: return concurrence, purity, min_eig, params[0], params[1], params[2], params[3]

def build_dataset(random_method, return_prob, num_to_gen, savename, do_stokes=False, include_w=True, log_params = False, log_roik_prob = False, verbose=False):
    ''' Fuction to build a dataset of randomly generated states.
    params:
        random_method: string, either 'simplex', 'jones_I','jones_C' or 'random'
        return_prob: bool, whether to return all 36 probabilities or 15 stokes's parameters
        savename: name of file to save data to
        do_stokes: bool, whether to use stokes's params to calc witnesses or operator
        include_w: bool, whether to include witness values in dataset
        log_params: bool, whether to collect the parameters that make up the 3rd unitary
        log_roik_prob: bool, whether to collect the probabilities using the roik et al definition
        verbose: bool, whether to print progress
    '''

    # confirm valid random method
    # assert random_method in ['simplex', 'jones_I','jones_C', 'hurwitz'], f'Invalid random method. You have {random_method}.'

    ## initialize dataframe to hold states ##
    # because of my Hoppy model in jones.py, we no longer need to save the generating angles; we can determine them!! :)) This doesn't work with mixed states, but we can still get a closest aproximation.

    if include_w:
        if not(log_params):
            if not(return_prob):
                columns = ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity', 'min_eig']
            elif return_prob and not(log_roik_prob):
                columns = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity', 'min_eig']
            else:
                columns = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL', 'r_HH', 'r_HV', 'r_HD', 'r_HA', 'r_HR', 'r_HL', 'r_VH', 'r_VV', 'r_VD', 'r_VA', 'r_VR', 'r_VL', 'r_DH', 'r_DV', 'r_DD', 'r_DA', 'r_DR', 'r_DL', 'r_AH', 'r_AV', 'r_AD', 'r_AA', 'r_AR', 'r_AL', 'r_RH', 'r_RV', 'r_RD', 'r_RA', 'r_RR', 'r_RL', 'r_LH', 'r_LV', 'r_LD', 'r_LA', 'r_LR', 'r_LL', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity', 'min_eig']
        else:
            if not(return_prob):
                columns = ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity', 'min_eig', 'alpha', 'psi', 'chi', 'phi']
            elif return_prob and not(log_roik_prob):
                columns = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity', 'min_eig', 'alpha', 'psi', 'chi', 'phi']
            else:
                columns = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL', 'r_HH', 'r_HV', 'r_HD', 'r_HA', 'r_HR', 'r_HL', 'r_VH', 'r_VV', 'r_VD', 'r_VA', 'r_VR', 'r_VL', 'r_DH', 'r_DV', 'r_DD', 'r_DA', 'r_DR', 'r_DL', 'r_AH', 'r_AV', 'r_AD', 'r_AA', 'r_AR', 'r_AL', 'r_RH', 'r_RV', 'r_RD', 'r_RA', 'r_RR', 'r_RL', 'r_LH', 'r_LV', 'r_LD', 'r_LA', 'r_LR', 'r_LL', 'W_min', 'Wp_t1', 'Wp_t2', 'Wp_t3', 'concurrence', 'purity', 'min_eig', 'alpha', 'psi', 'chi', 'phi']
    else:
        if not(log_params): columns = ['concurrence', 'purity', 'min_eig']
        else: columns = ['concurrence', 'purity', 'min_eig', 'alpha', 'psi', 'chi', 'phi']

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
        func = partial(get_random_hurwitz, method=method, log_params=log_params)
    elif random_method=='roik': # from actual roik code
        func = partial(get_random_rho, log_params=log_params)

    # build multiprocessing pool ##
    pool = Pool(cpu_count())
    # gen_rand_info(func, return_prob, do_stokes=False, include_w = True, log_params = False, verbose=True)
    inputs = [(func, return_prob, do_stokes, include_w, log_params, verbose, log_roik_prob) for _ in range(num_to_gen)]
    results = pool.starmap_async(gen_rand_info, inputs).get()

    ## end multiprocessing ##
    pool.close()
    pool.join()

    # filter None results out
    results = [result for result in results if result is not None] 

    # build df
    df = pd.DataFrame.from_records(results, columns = columns)
    print('saving!')
    df.to_csv(savename+'.csv', index=False)

def comp_me_roik():
    ''' Function to compare the states I generate using Roik's method to the ones generated with my method. 
    '''
    num_to_gen = int(input('How many states to generate? '))
    run_me = bool(int(input('Run my or Roik code? (1 or 0): ')))
    incude_w = bool(int(input('Include W? (1 or 0): ')))
    if run_me:
        savename = join('random_gen', 'test', f'6_27_me_{num_to_gen}')
        rtype='hurwitz'
    else:
        savename = join('random_gen', 'test', f'6_27_roik_{num_to_gen}')
        rtype='roik'

    build_dataset(rtype, True, num_to_gen, savename, include_w=incude_w, log_params=True)



## ask user for info ##
if __name__=='__main__':
    # random_method, return_prob, num_to_gen, savename
    do_roik = bool(int(input('Custom (0) or Compare with actual Roik (1): ')))
    if do_roik:
        comp_me_roik()
    else:
        random_method = input("Enter random method: 'simplex', 'jones_I','jones_C', 'random', 'hurwitz', or 'roik': ")
        return_prob = bool(int(input("Return probabilities (1) or stokes's parameters (0): ")))
        do_stokes = bool(int(input("Do stokes's parameters (1) or operators (0) to calculate witnesses: ")))
        num_to_gen = int(input("Enter number of states to generate: "))
        special = input("Enter special name for file: ")
        datadir = bool(int(input('Put in data dir (1) or test dir (0): ')))
        log_params = bool(int(input('Log parameters (1) or not (0): ')))
        log_roik_prob = bool(int(input('log roik prob? 0, 1: ')))

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

        print(f'{random_method}, {return_prob}, {num_to_gen}, {special}, {do_stokes}, {log_params}, {log_roik_prob}')

        build_dataset(random_method, return_prob, num_to_gen, savename, do_stokes=do_stokes, include_w=True, log_params=log_params, log_roik_prob=log_roik_prob)
        # random_method, return_prob, num_to_gen, savename, do_stokes=False, include_w=True, log_params = False, log_roik_prob = False, verbose=False
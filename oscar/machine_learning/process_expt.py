# file to read and process experimentally collected density matrices
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm

from sample_rho import *
from rho_methods import *

# set path
current_path = dirname(abspath(__file__))
DATA_PATH = join(current_path, '../../framework/decomp_test/')

def get_rho_from_file_depricated(filename, rho_actual):
    '''Function to read in experimental density matrix from file. Depricated since newer experiments will save the target density matrix in the file; for trials <= 14'''
    # read in data
    try:
        rho, unc, Su = np.load(join(DATA_PATH,filename), allow_pickle=True)
    except:
        rho, unc = np.load(join(DATA_PATH,filename), allow_pickle=True)

    # print results
    print('measured rho\n---')
    print(rho)
    print('uncertainty \n---')
    print(unc)
    print('actual rho\n ---')
    print(rho_actual)
    print('fidelity', get_fidelity(rho, rho_actual))
    print('purity', get_purity(rho))

    print('trace of measured rho', np.trace(rho))
    print('eigenvalues of measured rho', np.linalg.eigvals(rho))

def get_rho_from_file(filename, verbose=True):
    '''Function to read in experimental density matrix from file. For trials > 14. N.b. up to trial 23, angles were not saved (but recorded in lab_notebook markdown file). Also note that in trials 20 (E0 eta = 45), 21 (redo of E0 (eta = 45, chi = 0, 18)), 22 (E0 eta = 60), and 23 (E0 eta = 60, chi = -90), there was a sign error in the phi phase in the Jones matrices, so will recalculate the correct density matrix; ** the one saved in the file as the theoretical density matrix is incorrect **
    --
    Parameters
        filename : str, Name of file to read in
        verbose : bool, Whether to print out results
    '''

    # read in data
    try:
        rho, unc, Su, rho_actual, angles, fidelity, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)

        ## update df with info about this trial ##

        # print results
        if verbose:
            print('angles\n---')
            print(angles)
            print('measured rho\n---')
            print(rho)
            print('uncertainty \n---')
            print(unc)
            print('actual rho\n ---')
            print(rho_actual)
            print('fidelity', fidelity)
            print('purity', purity)

            print('trace of measured rho', np.trace(rho))
            print('eigenvalues of measured rho', np.linalg.eigvals(rho))
    except:
        rho, unc, Su, rho_actual, _, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)

        ## since angles were not saved, this means we also have the phi sign error as described in the comment to the function, so will need to recalculate the target. ##

        def split_filename():
            ''' Splits up the file name and identifies the trial number, eta, and chi values'''

            # split filename
            split_filename = filename.split('_')
            # get trial number
            trial = int(split_filename[-1].split('.')[0])
            # get eta
            eta = float(split_filename[1].split(',')[1].split('(')[1])
            chi = float(split_filename[1].split(',')[2].split(')')[0].split(' ')[1])

            return trial, eta, chi

        if "E0" in filename: # if E0, split up into eta and chi
            trial, eta, chi = split_filename()

            chi*=-1 # have to switch sign of chi

            # calculate target rho
            targ_rho = get_E0(np.deg2rad(eta), np.deg2rad(chi))
            fidelity = get_fidelity(rho, targ_rho)

            # print results
            if verbose:
                print('trial', trial)
                print('eta', eta)
                print('chi', chi)
                print('measured rho\n---')
                print(rho)
                print('uncertainty \n---')
                print(unc)

                print('actual rho\n ---')
                print(rho_actual)
                print('fidelity', fidelity)
                print('purity', purity)

            return trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi

        else: # if not E0, just print results
            trial = int(split_filename('.')[0].split('_')[-1])
            print('measured rho\n---')
            print(rho)
            print('uncertainty \n---')
            print(unc)
            print('actual rho\n ---')
            print(rho_actual)
            print('fidelity', fidelity)
            print('purity', purity)

            return trial, rho, unc, Su, rho_actual, fidelity, purity

def analyze_rhos(filenames):
    '''Extending get_rho_from_file to include multiple files; 
    __
    returns: df with:
        - trial number
        - eta (if they exist)
        - chi (if they exist)
        - fidelity
        - purity
        - W theory (adjusted for purity) and W expt and W unc
        - W' theory (adjusted for purity) and W' expt and W' unc
    '''
    # initialize df
    df = pd.DataFrame()

    for file in tqdm(filenames):
        try:
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi = get_rho_from_file(file, verbose=False)
        except:
            trial, rho, unc, Su, rho_actual, fidelity, purity = get_rho_from_file(file, verbose=False)
            eta, chi = None, None

        # calculate W and W' theory
        W_theory_ls = compute_witnesses(rho = rho_actual, expt_purity=purity)

        # calculate W and W' expt
        W_expt_ls = compute_witnesses(rho = rho, expt=True, stokes_unc=Su)

        # parse lists
        W_min_theory = W_theory_ls[0]
        Wp_t1_theory = W_theory_ls[1]
        Wp_t2_theory = W_theory_ls[2]
        Wp_t3_theory = W_theory_ls[3]
        # ---- #
        W_min_expt = W_expt_ls[0][0][0]
        W_min_unc = W_expt_ls[0][1][0]
        Wp_t1_expt = W_expt_ls[1][0]
        Wp_t1_unc = W_expt_ls[1][1]
        Wp_t2_expt = W_expt_ls[2][0]
        Wp_t2_unc = W_expt_ls[2][1]
        Wp_t3_expt = W_expt_ls[3][0]
        Wp_t3_unc = W_expt_ls[3][1]

        if eta is not None and chi is not None:
            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'eta':eta, 'chi':chi, 'fidelity':fidelity, 'purity':purity, 'W_min_AT':W_min_theory, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_theory, 'Wp_t2_AT':Wp_t2_theory, 'Wp_t3_AT':Wp_t3_theory, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc}])])

        else:
            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'fidelity':fidelity, 'purity':purity, 'W_min_AT':W_min_theory, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_theory, 'Wp_t2_AT':Wp_t2_theory, 'Wp_t3_AT':Wp_t3_theory, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc}])])

    # save df
    print('saving!')
    df.to_csv(join(DATA_PATH, 'rho_analysis.csv'))





 

if __name__ == '__main__':
    # set filenames for computing W values
    filenames_45 = ["rho_('E0', (45.0, 0.0))_20.npy", "rho_('E0', (45.0, 18.0))_20.npy", "rho_('E0', (45.0, 36.0))_20.npy", "rho_('E0', (45.0, 54.0))_20.npy", "rho_('E0', (45.0, 72.0))_20.npy", "rho_('E0', (45.0, 90.0))_20.npy"]
    filenames_60= ["rho_('E0', (59.99999999999999, 0.0))_22.npy", "rho_('E0', (59.99999999999999, 18.0))_22.npy", "rho_('E0', (59.99999999999999, 36.0))_22.npy", "rho_('E0', (59.99999999999999, 54.0))_22.npy", "rho_('E0', (59.99999999999999, 72.0))_22.npy", "rho_('E0', (59.99999999999999, 90.0))_22.npy"]
    filenames = filenames_45 + filenames_60

    # analyze rho files
    analyze_rhos(filenames)
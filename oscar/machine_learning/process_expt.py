# file to read and process experimentally collected density matrices
import numpy as np
from os.path import join, dirname, abspath

from sample_rho import *
from rho_methods import *

def get_rho_from_file_depricated(filename, rho_actual):
    '''Function to read in experimental density matrix from file. Depricated since newer experiments will save the target density matrix in the file; for trials <= 14'''
    # set path
    current_path = dirname(abspath(__file__))
    data_path = join(current_path, '../../framework/decomp_test/')
    # read in data
    try:
        rho, unc, Su = np.load(join(data_path,filename), allow_pickle=True)
    except:
        rho, unc = np.load(join(data_path,filename), allow_pickle=True)

    # print results
    print('measured rho\n---')
    print(rho)
    print('uncertainty \n---')
    print(unc)
    print('actual rho\n ---')
    print(rho_actual)
    print('fidelity', get_fidelity(rho, rho_actual))

    print('trace of measured rho', np.trace(rho))
    print('eigenvalues of measured rho', np.linalg.eigvals(rho))

def get_rho_from_file(filename):
    '''Function to read in experimental density matrix from file. For trials > 14'''
    # set path
    current_path = dirname(abspath(__file__))
    data_path = join(current_path, '../../framework/decomp_test/')
    # read in data
    try:
        rho, unc, Su, rho_actual, angles, fidelity, purity = np.load(join(data_path,filename), allow_pickle=True)

        ## update df with info about this trial ##

        # print results
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
        rho, unc, Su, rho_actual, fidelity, purity = np.load(join(data_path,filename), allow_pickle=True)

        ## update df with info about this trial ##

        # print results
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


get_rho_from_file("rho_('PhiP',)_19.npy")
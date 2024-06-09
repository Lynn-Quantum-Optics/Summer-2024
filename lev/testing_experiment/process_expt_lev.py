# file to read and process experimentally collected density matrices
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

# set path
current_path = dirname(abspath(__file__))
DATA_PATH = 'mixed_phi_psi_45'

def get_rho_from_file_depricated(filename, rho_actual):
    '''Function to read in experimental density matrix from file. Depricated since newer experiments will save the target density matrix in the file; for trials <= 14'''
    # read in data
    try:
        rho, unc, Su = np.load(join(DATA_PATH,filename), allow_pickle=True)
    except:
        rho, unc = np.load(join(DATA_PATH,filename), allow_pickle=True)

    # print results
    print('measublue rho\n---')
    print(rho)
    print('uncertainty \n---')
    print(unc)
    print('actual rho\n ---')
    print(rho_actual)
    print('fidelity', get_fidelity(rho, rho_actual))
    print('purity', get_purity(rho))

    print('trace of measublue rho', np.trace(rho))
    print('eigenvalues of measublue rho', np.linalg.eigvals(rho))

def get_rho_from_file(filename, verbose=True, angles=None):
    '''Function to read in experimental density matrix from file. For trials > 14. N.b. up to trial 23, angles were not saved (but recorded in lab_notebook markdown file). Also note that in trials 20 (E0 eta = 45), 21 (blueo of E0 (eta = 45, chi = 0, 18)), 22 (E0 eta = 60), and 23 (E0 eta = 60, chi = -90), there was a sign error in the phi phase in the Jones matrices, so will recalculate the correct density matrix; ** the one saved in the file as the theoretical density matrix is incorrect **
    --
    Parameters
        filename : str, Name of file to read in
        verbose : bool, Whether to print out results
        angles: list, List of angles used in the experiment. If not None, will assume angles provided in the data file.
    '''
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

    # read in data
    try:

        # rho, unc, Su, rho_actual, angles, fidelity, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
        rho, unc, Su, un_proj, un_proj_unc, _, angles, fidelity, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
    
        ## update df with info about this trial ##
        if "E0" in filename: # if E0, split up into eta and chi
            trial, eta, chi = split_filename()
        
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

            print('trace of measublue rho', np.trace(rho))
            print('eigenvalues of measublue rho', np.linalg.eigvals(rho))

        return trial, rho, unc, Su, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc
    except:
        rho, unc, Su, rho_actual, _, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
        # print(np.load(join(DATA_PATH,filename), allow_pickle=True))
        
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

            # chi*=-1 # have to switch sign of chi

            # calculate target rho
            targ_rho = get_E0(np.deg2rad(eta), np.deg2rad(chi))
            fidelity = get_fidelity(rho, targ_rho)

            # print results
            if verbose:
                print('trial', trial)
                print('eta', eta)
                print('chi', chi)
                print('measublue rho\n---')
                print(rho)
                print('uncertainty \n---')
                print(unc)

                print('actual rho\n ---')
                print(rho_actual)
                print('fidelity', fidelity)
                print('purity', purity)

            return trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles

        else: # if not E0, just print results
            trial = int(split_filename('.')[0].split('_')[-1])
            print('measublue rho\n---')
            print(rho)
            print('uncertainty \n---')
            print(unc)
            print('actual rho\n ---')
            print(rho_actual)
            print('fidelity', fidelity)
            print('purity', purity)

            return trial, rho, unc, Su, rho_actual, fidelity, purity, angles

def analyze_rhos(filenames, rho_actuals, settings=None, id='id'):
    '''Extending get_rho_from_file to include multiple files; 
    __
    inputs:
        filenames: list of filenames to analyze
        settings: dict of settings for the experiment
        id: str, special identifier of experiment; used for naming the df
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

    for i, file in tqdm(enumerate(filenames)):
        if settings is None:
            try:
                trial, rho, unc, Su, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
                #display('expt rho:', rho)
            except:
                trial, rho, unc, Su, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
                eta, chi = None, None
        else:
            try:
                trial, rho, _, Su, fidelity, purity, eta, chi, angles = get_rho_from_file(file, angles = settings[i], verbose=False)
            except:
                trial, rho, _, Su, fidelity, purity, angles = get_rho_from_file(file, verbose=False,angles=settings[i] )
                eta, chi = None, None
        rho_actual = rho_actuals[i]
        #display('theo rho:', rho_actual)
        # calculate W and W' theory
        W_T_ls = compute_witnesses(rho = rho_actual, verbose = True, return_params = True) # theory
        W_AT_ls = compute_witnesses(rho = adjust_rho(rho_actual, [eta, chi], 0.945), verbose = True) # adjusted theory

        # calculate W and W' expt
        W_expt_ls = compute_witnesses(rho = rho, expt=True, counts=unp.uarray(un_proj, un_proj_unc), verbose = True, return_params = True)

        # parse lists
        W_min_T = W_T_ls[0]
        Wp_t1_T = W_T_ls[1]
        Wp_t2_T = W_T_ls[2]
        Wp_t3_T = W_T_ls[3]
        W_name_T = W_T_ls[4]
        Wp1_name_T = W_T_ls[5]
        Wp2_name_T = W_T_ls[6]
        Wp3_name_T = W_T_ls[7]
        W_param_T = W_T_ls[8]
        Wp1_param_T = W_T_ls[9]
        Wp2_param_T = W_T_ls[10]
        Wp3_param_T = W_T_ls[11]
        # ---- #
        # not returning params for adjusted theory at the moment
        W_min_AT = W_AT_ls[0]
        Wp_t1_AT = W_AT_ls[1]
        Wp_t2_AT = W_AT_ls[2]
        Wp_t3_AT = W_AT_ls[3]
        W_name_AT = W_AT_ls[4]
        Wp1_name_AT = W_AT_ls[5]
        Wp2_name_AT = W_AT_ls[6]
        Wp3_name_AT = W_AT_ls[7]
        
        # ---- #
        # using propogated uncertainty
        try: # handle the observed difference in python 3.9.7 and 3.10
            W_min_expt = unp.nominal_values(W_expt_ls[0][0][0])
            W_min_unc = unp.std_devs(W_expt_ls[0][0][0])
        except: 
            W_min_expt = unp.nominal_values(W_expt_ls[0][0])
            W_min_unc = unp.std_devs(W_expt_ls[0][0])
        Wp_t1_expt = unp.nominal_values(W_expt_ls[1])
        Wp_t1_unc = unp.std_devs(W_expt_ls[1])
        Wp_t2_expt = unp.nominal_values(W_expt_ls[2])
        Wp_t2_unc = unp.std_devs(W_expt_ls[2])
        Wp_t3_expt = unp.nominal_values(W_expt_ls[3])
        Wp_t3_unc = unp.std_devs(W_expt_ls[3])
        W_name_expt = W_expt_ls[4]
        Wp1_name_expt = W_expt_ls[5]
        Wp2_name_expt = W_expt_ls[6]
        Wp3_name_expt = W_expt_ls[7]
        W_param_expt = W_expt_ls[8]
        Wp1_param_expt = W_expt_ls[9]
        Wp2_param_expt = W_expt_ls[10]
        Wp3_param_expt = W_expt_ls[11]
    
        if i == 2: # print out W values and rho for greatest differences W prime.
            print('At third data point, theo rho was:', rho_actual)
            print('And expt rho was:', rho)
            print('THE CURRENT POINT IS POINT:', i)
            print('With W/W primes of:', W_name_T, W_min_T, W_param_T, Wp1_name_T, Wp_t1_T, Wp1_param_T, Wp2_name_T, Wp_t2_T, Wp2_param_T, Wp3_name_T, Wp_t3_T, Wp3_param_T)
            print('Experimental Ws:', W_name_expt, W_min_expt, W_param_expt, Wp1_name_expt, Wp_t1_expt, Wp1_param_expt, Wp2_name_expt, Wp_t2_expt, Wp2_param_expt, Wp3_name_expt, Wp_t3_expt, Wp3_param_expt)

        if eta is not None and chi is not None:
            adj_fidelity= get_fidelity(adjust_rho(rho_actual, [eta, chi], purity), rho)

            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'eta':eta, 'chi':chi, 'fidelity':fidelity, 'purity':purity, 'AT_fidelity':adj_fidelity,
            'W_min_T': W_min_T, 'Wp_t1_T':Wp_t1_T, 'Wp_t2_T':Wp_t2_T, 'Wp_t3_T':Wp_t3_T,'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

        else:
            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'fidelity':fidelity, 'purity':purity, 'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

    # save df
    print('saving!')
    df.to_csv(join(DATA_PATH, f'analysis_{id}.csv'))

def make_plots_E0(dfname):
    '''Reads in df generated by analyze_rhos and plots witness value comparisons as well as fidelity and purity
    __
    dfname: str, name of df to read in
    num_plots: int, number of separate plots to make (based on eta)
    '''

    id = dfname.split('.')[0].split('_')[-1] # extract identifier from dfname

    # read in df
    df = pd.read_csv(join(DATA_PATH, dfname))
    eta_vals = df['eta'].unique()

    # preset plot sizes
    if len(eta_vals) == 1:
        fig, ax = plt.subplots(figsize = (10, 10))
        # get df for each eta
        df_eta = df
        purity_eta = df_eta['purity'].to_numpy()
        fidelity_eta = df_eta['fidelity'].to_numpy()
        chi_eta = df_eta['chi'].to_numpy()
        adj_fidelity = df_eta['AT_fidelity'].to_numpy()

        # # do purity and fidelity plots
        # ax[1,i].scatter(chi_eta, purity_eta, label='Purity', color='gold')
        # ax[1,i].scatter(chi_eta, fidelity_eta, label='Fidelity', color='turquoise')

        # # plot adjusted theory purity
        # ax[1,i].plot(chi_eta, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')

        # extract witness values
        W_min_T = df_eta['W_min_T'].to_numpy()
        W_min_AT = df_eta['W_min_AT'].to_numpy()
        print('In plotting, W_min_AT was:', W_min_AT)
        W_min_expt = df_eta['W_min_expt'].to_numpy()
        W_min_unc = df_eta['W_min_unc'].to_numpy()

        Wp_T = df_eta[['Wp_t1_T', 'Wp_t2_T', 'Wp_t3_T']].min(axis=1).to_numpy()
        Wp_AT = df_eta[['Wp_t1_AT', 'Wp_t2_AT', 'Wp_t3_AT']].min(axis=1).to_numpy()
        Wp_expt = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].min(axis=1).to_numpy()
        Wp_expt_min = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].idxmin(axis=1)
        Wp_unc = np.where(Wp_expt_min == 'Wp_t1_expt', df_eta['Wp_t1_unc'], np.where(Wp_expt_min == 'Wp_t2_expt', df_eta['Wp_t2_unc'], df_eta['Wp_t3_unc']))

        # plot curves for T and AT
        def sinsq(x, a, b, c, d):
            return a*np.sin(b*np.deg2rad(x) + c)**2 + d

        #print('W_min_T is:', W_min_T)
        #print('W_min_expt is:', W_min_expt)
        print('W min T is:', W_min_T)
        print('W min AT is:', W_min_AT)
        popt_W_T_eta, pcov_W_T_eta = curve_fit(sinsq, chi_eta, W_min_T)
        popt_W_AT_eta, pcov_W_AT_eta = curve_fit(sinsq, chi_eta, W_min_AT)
        #print('popt_W are:', popt_W_AT_eta) 
        popt_Wp_T_eta, pcov_Wp_T_eta = curve_fit(sinsq, chi_eta, Wp_T)
        popt_Wp_AT_eta, pcov_Wp_AT_eta = curve_fit(sinsq, chi_eta, Wp_AT)
        
        chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
        ax.errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue')

        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
        ax.errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon')
        ax.set_title(f'$\eta = 45\degree$', fontsize=18)
        ax.set_ylabel('Witness value', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(ncol=2, fontsize=16)
        ax.set_xlabel('$\chi$', fontsize=16)
        # ax[1,i].set_ylabel('Value', fontsize=31)
        # ax[1,i].legend()
    else:
        if len(eta_vals) == 2:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        elif len(eta_vals) == 3:
            fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex=True)
        
        for i, eta in enumerate(eta_vals):
            # get df for each eta
            df_eta = df[df['eta'] == eta]
            purity_eta = df_eta['purity'].to_numpy()
            fidelity_eta = df_eta['fidelity'].to_numpy()
            chi_eta = df_eta['chi'].to_numpy()
            adj_fidelity = df_eta['AT_fidelity'].to_numpy()

            # # do purity and fidelity plots
            # ax[1,i].scatter(chi_eta, purity_eta, label='Purity', color='gold')
            # ax[1,i].scatter(chi_eta, fidelity_eta, label='Fidelity', color='turquoise')

            # # plot adjusted theory purity
            # ax[1,i].plot(chi_eta, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')

            # extract witness values
            W_min_T = df_eta['W_min_T'].to_numpy()
            W_min_AT = df_eta['W_min_AT'].to_numpy()
            W_min_expt = df_eta['W_min_expt'].to_numpy()
            W_min_unc = df_eta['W_min_unc'].to_numpy()

            Wp_T = df_eta[['Wp_t1_T', 'Wp_t2_T', 'Wp_t3_T']].min(axis=1).to_numpy()
            Wp_AT = df_eta[['Wp_t1_AT', 'Wp_t2_AT', 'Wp_t3_AT']].min(axis=1).to_numpy()
            Wp_expt = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].min(axis=1).to_numpy()
            Wp_expt_min = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].idxmin(axis=1)
            Wp_unc = np.where(Wp_expt_min == 'Wp_t1_expt', df_eta['Wp_t1_unc'], np.where(Wp_expt_min == 'Wp_t2_expt', df_eta['Wp_t2_unc'], df_eta['Wp_t3_unc']))

            # plot curves for T and AT
            def sinsq(x, a, b, c, d):
                return a*np.sin(b*np.deg2rad(x) + c)**2 + d

            popt_W_T_eta, pcov_W_T_eta = curve_fit(sinsq, chi_eta, W_min_T)
            popt_W_AT_eta, pcov_W_AT_eta = curve_fit(sinsq, chi_eta, W_min_AT)

            popt_Wp_T_eta, pcov_Wp_T_eta = curve_fit(sinsq, chi_eta, Wp_T)
            popt_Wp_AT_eta, pcov_Wp_AT_eta = curve_fit(sinsq, chi_eta, Wp_AT)

            chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
            ax[i].errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue')


            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
            ax[i].errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon')

            ax[i].set_title('$\eta = 45\degree$', fontsize=33)
            ax[i].set_ylabel('Witness value', fontsize=31)
            ax[i].tick_params(axis='both', which='major', labelsize=25)
            ax[i].legend(ncol=2, fontsize=25)
            ax[i].set_xlabel('$\chi$', fontsize=31)
            # ax[1,i].set_ylabel('Value', fontsize=31)
            # ax[1,i].legend()
            
    plt.suptitle('Witnesses for Phi, Psi Bell Mix at 65/35 Probability', fontsize=22)
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'{id}.pdf'))
    plt.show()

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

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

# def get_theo_rho(alpha, beta):

#     H = ket([1,0])
#     V = ket([0,1])
    
#     PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
#     PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)

#     phi = np.cos(alpha)*PHI_PLUS + np.exp(1j*beta)*np.sin(alpha)*PHI_MINUS

#     rho = phi @ phi.conj().T

#     #rho = create_noise(rho, 2)
#     return rho

def gen_mixed_state(state_list, state_prob, eta_chi):
    '''
    Uses above helper functions to generate a given mixed state
    
    Parameters:
    state_list (list): list of state names that are to be mixed, must match creatable state names above
    state_prob (list): probability of each state being mixed in state_list, must match index
    eta_chi (list): what eta and chi to use for each state, must match index
    
    Returns:
    rho: an NxN density matrix
    '''
    
    # get individual rho's per state in state_list, taking probability into account
    individual_rhos = []
    for i, state in enumerate(state_list):
        individual_rhos.append(state_prob[i] * get_theo_rho(state, *eta_chi))
    # sum all matrices in individual rhos
    rho = np.sum(individual_rhos, axis = 0)
    
    return rho

def get_theo_rho(state, eta, chi):
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
    
    ##  The following 2 states inspired the Ws
    
    if state == 'phi plus, phi minus':
        phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PHI_MINUS
    
    if state == 'psi plus, psi minus':
        phi = np.cos(eta)*PSI_PLUS + np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    ## The following 6 states inspired the W primes
    
    # create the state PHI+ + PSI-
    if state == 'phi plus, psi minus':
        phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    # create the state PHI- + PSI+
    if state == 'phi minus, psi plus':
        phi = np.cos(eta)*PHI_MINUS + np.exp(1j*chi)*np.sin(eta)*PSI_PLUS
    
    # create the state PHI+ +i PSI+
    if state == 'phi plus, i psi plus':
        phi = np.cos(eta)*PHI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_PLUS
    
    # create the state PHI+ + i PHI-
    if state == 'phi plus, i phi minus':
        phi = np.cos(eta)*PHI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*PHI_MINUS
    
    # create the state PSI+ + iPSI-
    if state == 'psi plus, i psi minus':
        phi = np.cos(eta)*PSI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    if state == 'phi minus, i psi minus':
        phi = np.cos(eta)*PHI_MINUS + 1j*np.exp(1j*chi)*np.sin(eta)*PSI_MINUS
    
    # create rho and return it
    rho = phi @ phi.conj().T
    return rho

if __name__ == '__main__':
    # set filenames for computing W values

    etas = [np.pi/4]
    chis = np.linspace(0.001, np.pi/2, 6)
    states_names = []
    states = []
    # names = ['phi plus, psi minus', 'phi minus, psi plus']
    # probs = [0.65, 0.35]
    names = ['phi plus, phi minus', 'psi plus, psi minus']
    probs = [0.65, 0.35]
    
    for eta in etas:
        for chi in chis:
            states_names.append((np.rad2deg(eta), np.rad2deg(chi)))
            states.append((eta, chi))

    filenames = []
    settings = []
    rho_actuals = []
    # get file names for data produced from mix_expt_data
    for i, state_n in enumerate(states_names):
        filenames.append(f"rho_('E0', {state_n})_27.npy")
        settings.append([state_n[0],state_n[1]])

     # Obtain the density matrix for each state
    rho_actuals = []
    for i, state_set in enumerate(states_names):
        rad_angles = states[i]
        rho_actuals.append(gen_mixed_state(names, probs, rad_angles))

    # analyze rho files
    id = 'rho_phi-psi-bell'
    analyze_rhos(filenames, rho_actuals, id=id)
    make_plots_E0(f'analysis_{id}.csv')


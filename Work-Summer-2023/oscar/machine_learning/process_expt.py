# file to read and process experimentally collected density matrices
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, approx_fprime
from os.path import join, dirname, abspath, isdir
import pandas as pd

from tqdm import tqdm
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from uncertainties import ufloat
from uncertainties import unumpy as unp

from sample_rho import *
from rho_methods import *
from jones import *

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

def comp_rho_rhoadj(file="rho_('E0', (59.99999999999999, 90.0))_32.npy", UV_HWP_offset=1.029, special='default', model=None,  do_W = False, path = '../../framework/decomp_test'):
    '''Reads in experimental file and prints experimetnal rho with corrected'''
    r = np.load(join(path, file), allow_pickle=True)
    rho = r[0]
    # split filename
    split_filename = file.split('_')
    # get trial number
    trial = int(split_filename[-1].split('.')[0])
    # get eta
    eta = float(split_filename[1].split(',')[1].split('(')[1])
    chi = float(split_filename[1].split(',')[2].split(')')[0].split(' ')[1])
    print(f'trial: {trial}, eta: {eta}, chi: {chi}')
    angles = r[-3]
    print('angles before', angles)
    angles[0]+=UV_HWP_offset
    print('angles after', angles)
    rho_actual = get_Jrho(np.deg2rad(angles), setup='C0')
    purity = r[-1]

    # update eta and chi after changing the UVHWP angle
    print('eta, chi before:', eta, chi)
    eta_o, chi_o = eta, chi # save original angles
    eta, chi = correct_angles(angles, UV_HWP_offset, eta, chi)
    print('eta, chi after:', eta, chi)

    # rho_adj =  load_saved_get_E0_rho_c(rho_actual, [eta, chi], purity, model, UV_HWP_offset = UV_HWP_offset, do_W=do_W)
    if model==None:
        rho_adj =  adjust_rho(rho_actual, [eta, chi], purity)
    else:
        rho_adj = load_saved_get_E0_rho_c(rho_actual, [eta_o, chi_o], purity, model)
    print(rho)
    print('---------')
    print(rho_adj)
    print('-------')
    W_expt = compute_witnesses(rho)
    print('fidelity of rho to rho actual, original rho actual', r[-2])
    print(f'fidelity of rho to rho actual: {get_fidelity(rho, rho_actual)}')
    print('fidelity of rho to rho adj: ', get_fidelity(rho, rho_adj))
    print(f'purity rho: {get_purity(rho)}')
    print(f'purity adj: {get_purity(rho_adj)}')
    print('rho witnesses', W_expt)
    print('rho act witness', compute_witnesses(rho_actual))
    print('rho adj witnesses', compute_witnesses(rho_adj))
    print('stokes params rho', get_expec_vals(rho))
    print('stokes params rho actual', get_expec_vals(rho_actual))
    print('stokes params rho_adj', get_expec_vals(rho_adj))

    # make triptych 
    # separate magntiude and phase for each density matrix
    rho_mag = np.abs(rho)
    rho_phase = np.angle(rho)
    rho_actual_mag = np.abs(rho_actual)
    rho_actual_phase = np.angle(rho_actual)
    rho_adj_mag = np.abs(rho_adj)
    rho_adj_phase = np.angle(rho_adj)
    # plot confusion matrices
    fig, ax = plt.subplots(2,3, figsize=(15,8))
    sns.heatmap(rho_mag, annot=True, ax=ax[0,0], cmap='cool', vmin=0, vmax=1)
    sns.heatmap(rho_actual_mag, annot=True, ax=ax[0,1], cmap='cool', vmin=0, vmax=1)
    sns.heatmap(rho_adj_mag, annot=True, ax=ax[0,2], cmap='cool', vmin=0, vmax=1)
    sns.heatmap(rho_phase, annot=True, ax=ax[1,0], cmap='cool', vmin=-np.pi, vmax=np.pi)
    sns.heatmap(rho_actual_phase, annot=True, ax=ax[1,1], cmap='cool', vmin=-np.pi, vmax=np.pi)
    sns.heatmap(rho_adj_phase, annot=True, ax=ax[1,2], cmap='cool', vmin=-np.pi, vmax=np.pi)
    ax[0,0].set_title('$\\rho_{Expt}$')
    ax[0,1].set_title('$\\rho_{Theory}$')
    ax[0,2].set_title('$\\rho_{AdjTheory}$')
    ax[0,0].set_ylabel('Magnitude')
    ax[1,0].set_ylabel('Phase')
    plt.tight_layout()
    plt.suptitle('$E_0, \eta = %.3g\degree, \chi = %.3g\degree$, %s'%(eta_o, chi_o, special))
    # give space for title
    plt.subplots_adjust(top=0.9)
    plt.savefig('../../framework/decomp_test/%.3g_%.3g_comp_%s.pdf'%(eta_o, chi_o, special))
    plt.show()

    # plot stokes params for each density matrix
    # get only real parts of stokes params
    rho_stokes = np.real(get_expec_vals(rho))
    rho_actual_stokes = np.real(get_expec_vals(rho_actual))
    rho_adj_stokes = np.real(get_expec_vals(rho_adj))
    # plot confusion matrices, bounds are -1 to 1
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    sns.heatmap(rho_stokes, annot=True, ax=ax[0], cmap='cool', vmin=-1, vmax=1)
    sns.heatmap(rho_actual_stokes, annot=True, ax=ax[1], cmap='cool', vmin=-1, vmax=1)
    sns.heatmap(rho_adj_stokes, annot=True, ax=ax[2], cmap='cool', vmin=-1, vmax=1)
    ax[0].set_title('$\\rho_{Expt}$')
    ax[1].set_title('$\\rho_{Theory}$')
    ax[2].set_title('$\\rho_{AdjTheory}$')
    ax[0].set_ylabel('Stokes Parameters')
    plt.tight_layout()
    plt.suptitle('$E_0, \eta = %.3g\degree, \chi = %.3g\degree$, %s'%(eta_o, chi_o, special))
    # give space for title
    plt.subplots_adjust(top=0.9)
    plt.savefig('../../framework/decomp_test/%.3g_%.3g_stokes_%s.pdf'%(eta_o, chi_o, special))
    
    # if model==1:
    #     # make plot of W and W' as a function of e
    #     e = np.linspace(0, 1, 10)
    #     W = np.zeros_like(e)
    #     Wp = np.zeros_like(e)
    #     for i in range(len(e)):
    #         rho_adj = adjust_E0_rho_general([e[i]], rho_actual, purity, eta, chi)
    #         W_ls = compute_witnesses(rho_adj)
    #         W[i] = W_ls[0]
    #         Wp[i] = min(W_ls[1:])
    #     plt.figure(figsize=(10,7))
    #     plt.plot(e, W, label='$W$', color='blue')
    #     plt.plot(e, Wp, label='$W\'$', color='gold')
    #     # plot the experimental value of W and W'
    #     plt.plot(e, np.ones_like(e)*W_expt[0], label='$W_{Expt}$', color='blue', linestyle='--')
    #     plt.plot(e, np.ones_like(e)*min(W_expt[1:]), label='$W\'_{Expt}$', color='gold', linestyle='--')
    #     plt.xlabel('$e$')
    #     plt.ylabel('Witness value')
    #     plt.legend()
    #     plt.title('$E_0, \eta = %.3g\degree, \chi = %.3g\degree$, %s'%(eta_o, chi_o, special))
    #     plt.savefig('../../framework/decomp_test/%.3g_%.3g_witnesses_%s.pdf'%(eta_o, chi_o, special))
    #     plt.show()




def get_rho_from_file(filename, verbose=True, angles=None, do_richard = False):
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
        rho, unc, Su, un_proj, un_proj_unc, rho_actual, angles, fidelity, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
    

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

        return trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc
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

def analyze_rhos(filenames, UV_HWP_offset, settings=None, id='id', model=None, do_W = False, do_richard = False):
    '''Extending get_rho_from_file to include multiple files; 
    __
    inputs:
        filenames: list of filenames to analyze
        UV_HWP_offset: how many degrees to offset UV_HWP since callibration is believed to be off for 
        settings: dict of settings for the experiment
        id: str, special identifier of experiment; used for naming the df
        model: which model to use for purity correction; if None, will use the default model
        do_W: bool, whether to use W loss function calc for purity correction
    __
    returns: df with:
        - trial number
        - eta (if they exist)
        - chi (if they exist)
        - fidelity
        - purity
        - W theory (Purity Corrected for purity) and W expt and W unc
        - W' theory (Purity Corrected for purity) and W' expt and W' unc
    '''
    # initialize df
    df = pd.DataFrame()

    for i, file in tqdm(enumerate(filenames)):
        if settings is None:
            try:
                trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            except:
                trial, rho, unc, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
                eta, chi = None, None
        else:
            try:
                trial, rho, _, Su, rho_actual, fidelity, purity, eta, chi, angles = get_rho_from_file(file, angles = settings[i], verbose=False)
            except:
                trial, rho, _, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False,angles=settings[i] )
                eta, chi = None, None

        angles[0]+=UV_HWP_offset
        rho_actual = get_Jrho(angles=np.deg2rad(angles))
        eta_c, chi_c = correct_angles(angles, UV_HWP_offset, eta, chi)
        rho_adj = load_saved_get_E0_rho_c(rho_actual = rho_actual, angles_save = [eta, chi], angles_cor = [eta_c, chi_c], purity = purity, model = model, UV_HWP_offset = UV_HWP_offset, do_W = do_W)

        # get the fixed fidelity based on the offset to UVHWP
        fidelity = get_fidelity(rho, rho_actual)

        # calculate W and W' theory
        W_T_ls = compute_witnesses(rho = rho_actual) # theory
        W_AT_ls = compute_witnesses(rho = rho_adj) # Purity Corrected theory

        # calculate W and W' expt
        W_expt_ls = compute_witnesses(rho = rho, expt=True, counts=unp.uarray(un_proj, un_proj_unc))
        # W_expt_ls = compute_witnesses(rho = rho)

        # parse lists
        W_min_T = W_T_ls[0]
        Wp_t1_T = W_T_ls[1]
        Wp_t2_T = W_T_ls[2]
        Wp_t3_T = W_T_ls[3]
        # ---- #
        W_min_AT = W_AT_ls[0]
        Wp_t1_AT = W_AT_ls[1]
        Wp_t2_AT = W_AT_ls[2]
        Wp_t3_AT = W_AT_ls[3]
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

        # if eta is not None and chi is not None:
        #     # if model is None:
        #     #     adj_rho = adjust_rho(rho_actual, [eta, chi], purity)
        #     #     adj_fidelity = get_fidelity(adj_rho, rho)
        #     #     adj_purity = get_purity(adj_rho)
        #     # else:
        #     #     adj_fidelity, adj_purity = get_adj_E0_fidelity_purity(rho, rho_actual, purity, eta, chi, model, UV_HWP_offset)
        #     if model is None:
        #         adj_rho = adjust_rho(rho_actual, [eta_c, chi_c], purity)
        #     else:
        #         adj_rho = load_saved_get_E0_rho_c(rho_actual = rho_actual, angles = [eta, chi], purity = purity, model = model, UV_HWP_offset = UV_HWP_offset, do_W = do_W)
        adj_fidelity = get_fidelity(rho_adj, rho)
        adj_purity = get_purity(rho_adj)

        df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'eta':eta, 'chi':chi, 'eta_c': eta_c, 'chi_c': chi_c, 'fidelity':fidelity, 'purity':purity, 'AT_fidelity':adj_fidelity, 'AT_purity': adj_purity,
        'W_min_T': W_min_T, 'Wp_t1_T':Wp_t1_T, 'Wp_t2_T':Wp_t2_T, 'Wp_t3_T':Wp_t3_T,'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

        # else:
        #     df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'fidelity':fidelity, 'purity':purity, 'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

    # save df
    print('saving!')
    if not(do_W):
        df.to_csv(join(DATA_PATH, f'rho_analysis_{id}.csv'))
    else:
        df.to_csv(join(DATA_PATH, f'rho_analysis_{id}_W.csv'))

def make_plots_E0(dfname, paper=False):
    '''Reads in df generated by analyze_rhos and plots witness value comparisons as well as fidelity and purity
    __
    dfname: str, name of df to read in
    num_plots: int, number of separate plots to make (based on eta)
    paper: bool, whether to make plots for paper or not
    '''

    id = dfname.split('.csv')[0] # extract identifier from dfname

    # read in df
    df = pd.read_csv(join(DATA_PATH, dfname))
    eta_vals = df['eta'].unique()

    # preset plot sizes
    if not paper:
        if len(eta_vals) == 2:
            fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
        elif len(eta_vals) == 3:
            fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex=True)
    else:
        if len(eta_vals) == 3:
            fig, ax = plt.subplots(3, 2, figsize=(14, 14), sharex=True)

    for i, eta in enumerate(eta_vals):
        # get df for each eta
        df_eta = df.loc[np.round(df['eta'], 4) == np.round(eta,3)]
        print(df_eta)
        purity_eta = df_eta['purity'].to_numpy()
        fidelity_eta = df_eta['fidelity'].to_numpy()
        chi_eta = df_eta['chi'].to_numpy()
        adj_fidelity = df_eta['AT_fidelity'].to_numpy()
        adj_purity = df_eta['AT_purity'].to_numpy()

        # do purity and fidelity plots
        if paper:
            ax[i, 1].scatter(chi_eta, purity_eta, label='Purity', color='gold')
            ax[i, 1].plot(chi_eta, adj_purity, color='gold', linestyle='dashed', label='AT Purity')
            ax[i, 1].scatter(chi_eta, fidelity_eta, label='Fidelity', color='turquoise')
            ax[i, 1].plot(chi_eta, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')
        else:
            ax[1,i].scatter(chi_eta, purity_eta, label='Purity', color='gold')
            ax[1,i].plot(chi_eta, adj_purity, color='gold', linestyle='dashed', label='AT Purity')
            ax[1,i].scatter(chi_eta, fidelity_eta, label='Fidelity', color='turquoise')
            ax[1,i].plot(chi_eta, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')

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

        def line(x, a, b):
            return a*x + b
        try: # if W is really close to 0, it will have hard time fitting sinusoid, so fit line instead
            popt_W_T_eta, pcov_W_T_eta = curve_fit(sinsq, chi_eta, W_min_T)
            popt_W_AT_eta, pcov_W_AT_eta = curve_fit(sinsq, chi_eta, W_min_AT)
        except:
            popt_W_T_eta, pcov_W_T_eta = curve_fit(line, chi_eta, W_min_T)
            popt_W_AT_eta, pcov_W_AT_eta = curve_fit(line, chi_eta, W_min_AT)
        try:
            popt_Wp_T_eta, pcov_Wp_T_eta = curve_fit(sinsq, chi_eta, Wp_T)
            popt_Wp_AT_eta, pcov_Wp_AT_eta = curve_fit(sinsq, chi_eta, Wp_AT)
        except:
            popt_Wp_T_eta, pcov_Wp_T_eta = curve_fit(line, chi_eta, Wp_T)
            popt_Wp_AT_eta, pcov_Wp_AT_eta = curve_fit(line, chi_eta, Wp_AT)

        chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

        if not paper:
            try:
                ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            except:
                ax[0,i].plot(chi_eta_ls, line(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            try:
                ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
            except:
                ax[0,i].plot(chi_eta_ls, line(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
            
            ax[0,i].errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue', label='$W_{expt}$')

            try:
                ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
            except:
                ax[0,i].plot(chi_eta_ls, line(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')

            try:
                ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
            except:
                ax[0,i].plot(chi_eta_ls, line(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')

            ax[0,i].errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon', label="$W_{expt}'$")

            ax[0,i].set_title(f'$\eta = {np.round(eta,3)}$')
            ax[0,i].set_ylabel('Witness value')
            ax[0,i].legend(ncol=3)
            ax[1,i].set_xlabel('$\chi$')
            ax[1,i].set_ylabel('Value')
            ax[1,i].legend()

        else: # same as above, but swap indices
            try: 
                ax[i, 0].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            except:
                ax[i, 0].plot(chi_eta_ls, line(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            try:
                ax[i, 0].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
            except:
                ax[i, 0].plot(chi_eta_ls, line(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
            
            ax[i,0].errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue', label='$W_{expt}$')

            try:
                ax[i,0].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
            except:
                ax[i,0].plot(chi_eta_ls, line(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')

            try:
                ax[i,0].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
            except:
                ax[i,0].plot(chi_eta_ls, line(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')

            ax[i,0].errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon', label="$W_{expt}'$")

            ax[i,0].set_title(f'$\eta = {np.round(eta,3)}$')
            ax[i,0].set_ylabel('Witness value')
            ax[i,0].legend(ncol=3)
            ax[i,1].set_xlabel('$\chi$')
            ax[i,1].set_ylabel('Value')
            ax[i,1].legend()

    plt.suptitle('Witnesses for $E_0$ states, $\cos(\eta)|\Psi^+\\rangle + \sin(\eta)e^{i \chi}|\Psi^-\\rangle $')
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'exp_witnesses_E0_{id}.pdf'))
    plt.show()

    # plot angle settings as a function of chi
    fig, ax = plt.subplots(len(eta_vals),3, figsize=(25,10))

    for i, eta in enumerate(eta_vals):
        # get df for each eta
        df_eta = df[df['eta'] == eta]
    
        UV_HWP = df_eta['UV_HWP'].to_numpy()
        QP = df_eta['QP'].to_numpy()
        B_HWP = df_eta['B_HWP'].to_numpy()

        # plot
        ax[i, 0].scatter(chi_eta, UV_HWP)
        ax[i, 1].scatter(chi_eta, QP)
        ax[i, 2].scatter(chi_eta, B_HWP)
        ax[i, 0].set_title(f'$\eta = {np.round(eta,3)}\degree$, UV HWP')
        ax[i, 0].set_ylabel('Angle (deg)')
        ax[i, 0].set_xlabel('$\chi$')
        ax[i, 1].set_title(f'$\eta = {np.round(eta,3)}\degree$, QP')
        ax[i, 1].set_ylabel('Angle (deg)')
        ax[i, 1].set_xlabel('$\chi$')
        ax[i, 2].set_title(f'$\eta = {np.round(eta,3)}\degree$, B HWP')
        ax[i, 2].set_ylabel('Angle (deg)')
        ax[i, 2].set_xlabel('$\chi$')
    plt.suptitle('Angle settings for $E_0$ states, $\cos(\eta)|\Psi^+\\rangle + \sin(\eta)e^{i \chi}|\Psi^-\\rangle $')
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'exp_angles_E0_{id}.pdf'))
    plt.show()

def correct_angles(angles, UV_HWP_offset, eta, chi):
    '''Corrects angles for UV HWP offset'''
    angles[0] += UV_HWP_offset
    
    # get jones matrix
    rho_th = get_Jrho(np.deg2rad(angles))
    # determine eta and chi
    eta_c, chi_c = eta, chi
    # use gd to determine angles
    def loss(eta_chi):
        eta, chi = eta_chi   
        rho = get_E0(np.deg2rad(eta), np.deg2rad(chi))
        return 1-np.sqrt(get_fidelity(rho, rho_th))

    res = minimize(loss, [eta_c, chi_c], method='Nelder-Mead')
    eta_c, chi_c = res.x
    print('fidelity for new eta and chi to theoretical', get_fidelity(get_E0(np.deg2rad(eta_c), np.deg2rad(chi_c)), rho_th))
    return eta_c, chi_c

def comp_w_adj(filenames, UV_HWP_offset=1.029, id='id', model=1, do_W = False, do_richard = False):
    '''Function to compare the witness values for experimental to theoretical and adjusted theoreitcal'''
    # store W, Wp_t1, Wp_t2, and Wp_t3 for each eta and chi for experimental, theoretical, and adjusted theoretical in separate lists
    W_expt, Wp_t1_expt, Wp_t2_expt, Wp_t3_expt = [], [], [], []
    W_theo, Wp_t1_theo, Wp_t2_theo, Wp_t3_theo = [], [], [], []
    W_adj, Wp_t1_adj, Wp_t2_adj, Wp_t3_adj = [], [], [], []
    eta_ls, chi_ls = [], []

    # read in files
    for i, file in tqdm(enumerate(filenames)):
        try:
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
        except:
            trial, rho, unc, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
            eta, chi = None, None
    
        angles[0]+=UV_HWP_offset
        rho_actual = get_Jrho(angles=np.deg2rad(angles))
        eta_c, chi_c = correct_angles(angles, UV_HWP_offset, eta, chi)
        rho_adj = load_saved_get_E0_rho_c(rho_actual = rho_actual, angles_save = [eta, chi], angles_cor = [eta_c, chi_c], purity = purity, model = model, UV_HWP_offset = UV_HWP_offset, do_W = do_W)

        # calculate witness values
        W_expt_witnesses = compute_witnesses(rho)
        W_theo_witnesses = compute_witnesses(rho_actual)
        W_adj_witnesses = compute_witnesses(rho_adj)

        # add to lists
        W_expt.append(W_expt_witnesses[0])
        Wp_t1_expt.append(W_expt_witnesses[1])
        Wp_t2_expt.append(W_expt_witnesses[2])
        Wp_t3_expt.append(W_expt_witnesses[3])

        W_theo.append(W_theo_witnesses[0])
        Wp_t1_theo.append(W_theo_witnesses[1])
        Wp_t2_theo.append(W_theo_witnesses[2])
        Wp_t3_theo.append(W_theo_witnesses[3])

        W_adj.append(W_adj_witnesses[0])
        Wp_t1_adj.append(W_adj_witnesses[1])
        Wp_t2_adj.append(W_adj_witnesses[2])
        Wp_t3_adj.append(W_adj_witnesses[3])

        eta_ls.append(eta)
        chi_ls.append(chi)

    # put in a dataframe so we can separate by eta
    df = pd.DataFrame({'W_expt': W_expt, 'Wp_t1_expt': Wp_t1_expt, 'Wp_t2_expt': Wp_t2_expt, 'Wp_t3_expt': Wp_t3_expt, 'W_theo': W_theo, 'Wp_t1_theo': Wp_t1_theo, 'Wp_t2_theo': Wp_t2_theo, 'Wp_t3_theo': Wp_t3_theo, 'W_adj': W_adj, 'Wp_t1_adj': Wp_t1_adj, 'Wp_t2_adj': Wp_t2_adj, 'Wp_t3_adj': Wp_t3_adj, 'eta': eta_ls, 'chi': chi_ls})

    df.to_csv(f'../../framework/decomp_test/comp_w_adj_{model}.csv')

    # df= pd.read_csv(f'../../framework/decomp_test/comp_w_adj_{model}.csv')

    # separate by eta
    eta_unique = df['eta'].unique()
    fig, ax = plt.subplots(len(eta_unique), 4, figsize=(15, 10))
    for i, eta in enumerate(eta_unique):
        # get df for each eta
        df_eta = df[df['eta']==eta]
        W_expt = df_eta['W_expt']
        Wp_t1_expt = df_eta['Wp_t1_expt']
        Wp_t2_expt = df_eta['Wp_t2_expt']
        Wp_t3_expt = df_eta['Wp_t3_expt']
        W_theo = df_eta['W_theo']
        Wp_t1_theo = df_eta['Wp_t1_theo']
        Wp_t2_theo = df_eta['Wp_t2_theo']
        Wp_t3_theo = df_eta['Wp_t3_theo']
        W_adj = df_eta['W_adj']
        Wp_t1_adj = df_eta['Wp_t1_adj']
        Wp_t2_adj = df_eta['Wp_t2_adj']
        Wp_t3_adj = df_eta['Wp_t3_adj']
        chi = df_eta['chi']

        # plot
        ax[i, 0].plot(chi, W_expt, linestyle='solid', label='expt')
        ax[i, 0].plot(chi, W_theo, linestyle='dashed', label='theo')
        ax[i, 0].plot(chi, W_adj, linestyle='dashdot', label='adj')
        ax[i, 0].set_title('$\eta = {}$'.format(np.round(eta,3)))
        ax[i, 0].set_xlabel('$\chi$')
        ax[i, 0].set_ylabel('$W$')
        ax[i, 0].legend()
        ax[i, 1].plot(chi, Wp_t1_expt, linestyle='solid', label='expt')
        ax[i, 1].plot(chi, Wp_t1_theo, linestyle= 'dashed', label='theo')
        ax[i, 1].plot(chi, Wp_t1_adj, linestyle= 'dashdot', label='adj')
        ax[i, 1].set_title('$\eta = {}$'.format(np.round(eta,3)))
        ax[i, 1].set_xlabel('$\chi$')
        ax[i, 1].set_ylabel('$W\'_{t_1}$')
        ax[i, 1].legend()
        ax[i, 2].plot(chi, Wp_t2_expt, linestyle='solid', label='expt')
        ax[i, 2].plot(chi, Wp_t2_theo, linestyle= 'dashed', label='theo')
        ax[i, 2].plot(chi, Wp_t2_adj, linestyle= 'dashdot', label='adj')
        ax[i, 2].set_title('$\eta = {}$'.format(np.round(eta,3)))
        ax[i, 2].set_xlabel('$\chi$')
        ax[i, 2].set_ylabel('$W\'_{t_2}$')
        ax[i, 2].legend()

        ax[i, 3].plot(chi, Wp_t3_expt, linestyle='solid', label='expt')
        ax[i, 3].plot(chi, Wp_t3_theo, linestyle= 'dashed', label='theo')
        ax[i, 3].plot(chi, Wp_t3_adj, linestyle= 'dashdot', label='adj')
        ax[i, 3].set_title('$\eta = {}$'.format(np.round(eta,3)))
        ax[i, 3].set_xlabel('$\chi$')
        ax[i, 3].set_ylabel('$W\'_{t_3}$')
    plt.tight_layout()
    plt.suptitle('Model = {}'.format(model))
    # add space for title
    plt.subplots_adjust(top=0.92)
    plt.savefig('../../framework/decomp_test/comp_w_adj_{}.pdf'.format(model))
    plt.show()

def analyze_diff(filenames, adjust_file='noise_model.csv', settings=None):
    '''Compare difference of actual and experimental density matrices for each chi and eta
    --
    Params:
        filenames: list of files containing experimental data to analyze
        adjust_file: str, filename of file containing noise params
        settings: list of settings for the experiment, depricated

    '''
    diagonal1_mag = []
    diagonal1_mag_c = []
    diagonal1_mag_c2 = []
    anti_diagonal_mag = []
    anti_diagonal_phase = []
    anti_diagonal_mag_c = []
    anti_diagonal_phase_c = []
    anti_diagonal_mag_c2 = []
    anti_diagonal_phase_c2 = []
    UV_HWP_ls = []
    QP_ls = []
    B_HWP_ls = []

    adjust_df = pd.read_csv(join(DATA_PATH, adjust_file))

    for i, file in tqdm(enumerate(filenames)):
        print('----------')
        print(file)
        if settings is None:
            try:
                trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            except:
                trial, rho, unc, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
                eta, chi = None, None
        else:
            try:
                trial, rho, _, Su, rho_actual, fidelity, purity, eta, chi, angles = get_rho_from_file(file, angles = settings[i], verbose=False)
            except:
                trial, rho, _, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False,angles=settings[i] )
                eta, chi = None, None

        # get corrected rho
        rho_adj = adjust_rho(rho_actual, [eta, chi], purity)
        # adjust according to general noise
        eta_chi_adjust = adjust_df[(np.round(adjust_df['eta'],4) == np.round(eta, 4)) & (np.round(adjust_df['chi'], 4) == np.round(chi, 4))]

        eta_chi_adjust = eta_chi_adjust.iloc[0][['r_hh', 'r_hv', 'r_vh', 'r_vv']]
        eta_chi_adjust = eta_chi_adjust.to_numpy()
        eta_chi_adjust /= np.sum(eta_chi_adjust) # need to normalize
        # pass to rho adjust general
        print(eta_chi_adjust)
        rho_adj2 = adjust_rho_general(eta_chi_adjust, rho_actual, purity)
        
        
        
        # adjust components according to gd
        # angles_c = angles.copy()
        # angles_c[1] += 1.08
        # attempt 1 #
        # angles_c[0] += 0.5650933039044096
        # angles_c[1] += -0.6800269700198879
        # angles_c[2] += 0.82132298586235
        # attempt 2 #
        # angles_c[0] += -3.65158264
        # angles_c[1] += -3.65158264
        # angles_c[2] += 4.11424762
        # attempt 3 #
        # angles_c[0] += 4.291483685555388
        # angles_c[1] += -2.5321073619543957
        # angles_c[2] += -0.33779445243176065
        # angles_c[0] += 1.14862578203714
        # angles_c[1] += -1.229811759034985
        # angles_c[2] += -1.5514375868209096
        # angles_c[0] += 0.34932816157896734
        # angles_c[1] += 2.9149193113160328
        # angles_c[2] += 0.47171094035645134
        # angles_c[0] += 2.1708454661402072
        # angles_c[1] += 3.342929859171093
        # angles_c[2] += -3.7403145059563316
        # angles_c = np.deg2rad(angles_c)
        # rho_adj2 = get_Jrho(angles_c)
        # rho_adj2 = adjust_rho(rho_adj2, [eta, chi], purity)

        if i == 0: # set initial value of eta
            eta_0 = eta
            chi_ls = [chi]
            fidelity_ls = [fidelity]
            purity_ls = [purity]
            fidelity_adj_ls = [get_adj_fidelity(rho_actual, angles, purity)]
            purity_adj_ls = [get_purity(rho_adj)]
            fidelity_adj2_ls = [get_fidelity(rho_adj2, rho)]
            purity_adj2_ls = [get_purity(rho_adj2)]
            UV_HWP_ls = [angles[0]]
            QP_ls = [angles[1]]
            B_HWP_ls = [angles[2]]
        if eta != eta_0 or i==len(filenames)-1: # if different eta, reset chi_ls
            # plot the magnitudes and phase
            fig, ax = plt.subplots(2,4, figsize=(20,10), sharex=True)
            ax[0,0].scatter(chi_ls, diagonal1_mag, label='Actual', marker = '*', sizes = [100 for _ in chi_ls], color='red')
            ax[0,0].scatter(chi_ls, diagonal1_mag_c, label='Purity Corrected', color='green')
            ax[0,0].scatter(chi_ls, diagonal1_mag_c2, label='General Purity', color='blue')
            ax[0,0].set_title('Diagonal 1 Magnitude')
            ax[0,0].legend()
            ax[0,1].scatter(chi_ls, anti_diagonal_mag, label='Actual', marker = '*', sizes = [100 for _ in chi_ls], color='red')
            ax[0,1].scatter(chi_ls, anti_diagonal_mag_c, label='Purity Corrected', color='green')
            ax[0,1].scatter(chi_ls, anti_diagonal_mag_c2, label='General Purity', color='blue')
            ax[0,1].legend()
            ax[0,1].set_title('Anti-Diagonal Magnitude')
            ax[0,2].scatter(chi_ls, anti_diagonal_phase, label='Actual', marker = '*', sizes = [100 for _ in chi_ls], color='red')
            ax[0,2].scatter(chi_ls, anti_diagonal_phase_c, label='Purity Corrected', color='green')
            ax[0,2].scatter(chi_ls, anti_diagonal_phase_c2, label='General Purity', color='blue')
            ax[0,2].legend()
            ax[0,2].set_title('Anti-Diagonal Phase')
            ax[0,3].scatter(chi_ls, fidelity_ls, label='Actual', marker = '*', sizes = [100 for _ in chi_ls], color='red')
            ax[0,3].scatter(chi_ls, fidelity_adj_ls, label='Purity Corrected', color='green')
            ax[0,3].scatter(chi_ls, fidelity_adj2_ls, label='General Purity', color='blue')
            ax[0,3].legend()
            ax[0,3].set_title('Fidelity')
            ax[1,3].scatter(chi_ls, purity_ls, label='Actual', marker = '*', sizes = [100 for _ in chi_ls], color='red')
            ax[1,3].scatter(chi_ls, purity_adj_ls, label='Purity Corrected', color='green')
            ax[1,3].scatter(chi_ls, purity_adj2_ls, label='General Purity', color='blue')
            ax[1,3].legend()
            ax[1,3].set_title('Purity')
            ax[1,0].scatter(chi_ls, UV_HWP_ls)
            ax[1,0].set_title('UV HWP')
            ax[1,1].scatter(chi_ls, QP_ls)
            ax[1,1].set_title('QP')
            ax[1,2].scatter(chi_ls, B_HWP_ls)
            ax[1,2].set_title('B HWP')

            ax[0,0].set_ylabel('$\\frac{r_{\\rho_{Th}}}{r_{\\rho_{Expt}}}$')
            ax[0,1].set_ylabel('$\\frac{r_{\\rho_{Th}}}{r_{\\rho_{Expt}}}$')
            ax[0,2].set_ylabel('$\\frac{\\theta_{\\rho_{Th}}} {\\theta_{\\rho_{Expt}}}$')
            ax[1,0].set_xlabel('$\chi$')
            ax[1,1].set_xlabel('$\chi$')
            ax[1,2].set_xlabel('$\chi$')
            ax[1,3].set_xlabel('$\chi$')
            plt.suptitle(f'Differences for $\eta = {np.round(eta_0,3)}$')
            plt.tight_layout()
            plt.savefig(join(DATA_PATH, 'diff_r_phi', f'analysis_{eta_0}.pdf'))
            plt.close()
            

            # reset vals
            eta_0 = eta
            chi_ls = [chi]
            fidelity_ls = [fidelity]
            purity_ls = [purity]
            diagonal1_mag = []
            diagonal1_mag_c = []
            anti_diagonal_mag = []
            anti_diagonal_phase = []
            anti_diagonal_mag_c = []
            anti_diagonal_phase_c = []
            anti_diagonal_mag_c2 = []
            anti_diagonal_phase_c2 = []
            diagonal1_mag_c2 = []
            fidelity_adj_ls = [get_adj_fidelity(rho_actual, angles, purity)]
            purity_adj_ls = [get_purity(rho_adj)]
            fidelity_adj2_ls = [get_fidelity(rho_adj2, rho)]
            purity_adj2_ls = [get_purity(rho_adj2)]
            UV_HWP_ls = [angles[0]]
            QP_ls = [angles[1]]
            B_HWP_ls = [angles[2]]
            
        elif i!=0:
            chi_ls.append(chi)
            fidelity_ls.append(fidelity)
            purity_ls.append(purity)
            fidelity_adj_ls.append(get_adj_fidelity(rho_actual, angles, purity))
            purity_adj_ls.append(get_purity(rho_adj))
            fidelity_adj2_ls.append(get_fidelity(rho_adj2, rho))
            purity_adj2_ls.append(get_purity(rho_adj2))
            UV_HWP_ls.append(angles[0])
            QP_ls.append(angles[1])
            B_HWP_ls.append(angles[2])
        # take difference of actual and experimental density matrices
        phi_act = np.angle(rho_actual, deg=True)
        phi = np.angle(rho, deg=True)
        # get magnitude diff
        rho_actual_mag = np.abs(rho_actual)
        rho_mag = np.abs(rho)
        
        diff_r = rho_actual_mag / rho_mag
        diff_phi = phi_act  / phi

        # log diagonal magntidues and anti-diagonal magnitude
        # log antidiagonal phase
        diagonal1_mag.append(diff_r[1,1])
        anti_diagonal_mag.append(diff_r[1,2])
        anti_diagonal_phase.append(diff_phi[1,2])

        #### correction ####
        phi_adj = np.angle(rho_adj, deg=True)
        rho_adj_mag = np.abs(rho_adj)

        diff_r_adj = rho_adj_mag / rho_mag
        diff_phi_adj = phi_adj/ phi

        anti_diagonal_mag_c.append(diff_r_adj[1,2])
        anti_diagonal_phase_c.append(diff_phi_adj[1,2])
        diagonal1_mag_c.append(diff_r_adj[1,1])

        #### correction 2 ####
        phi_adj2 = np.angle(rho_adj2, deg=True)
        rho_adj_mag2 = np.abs(rho_adj2)

        diff_r_adj2 = rho_adj_mag2 / rho_mag
        diff_phi_adj2 = phi_adj2/ phi

        anti_diagonal_mag_c2.append(diff_r_adj2[1,2])
        anti_diagonal_phase_c2.append(diff_phi_adj2[1,2])
        diagonal1_mag_c2.append(diff_r_adj2[1,1])
        
        fig, ax = plt.subplots(1,2, figsize=(20,10))
        sns.heatmap(diff_r_adj2, cmap='coolwarm', annot=True, fmt='.2f', ax=ax[0])
        ax[0].set_title('Magnitude Ratio $\\frac{r_{\\rho_{Th}}}{r_{\\rho_{Expt}}}$')
        sns.heatmap(diff_phi_adj2, cmap='coolwarm', annot=True, fmt='.2f', ax=ax[1])

        ax[1].set_title('Phase Difference $\\frac{\\theta_{\\rho_{Th}}} {\\theta_{\\rho_{Expt}}}$')
        plt.suptitle(f'Matrix for $\eta = {np.round(eta,3)}, \chi={np.round(chi,3)}$')
        plt.tight_layout()
        plt.savefig(join(DATA_PATH, 'diff_r_phi', f'diff2_adj_{eta}_{chi}.pdf'))

def det_offsets(filenames, N=500, zeta=1, f=.02, loss_lim = 1e-9):
    '''Determine offsets in UV HWP, QP, and B HWP that minimize the loss function (sum of squares of fidelity differences)
    --
    Params:
    filenames: list of str, filenames of rho files to use
    N: int, number of iterations to run
    zeta: float, learning rate
    f: float, fraction of times to exit gd and do random
    loss_lim: float, loss limit to exit gd
    
    '''
    def get_new_fidelity(x0, angles, purity, eta, chi, rho):
        a, b, c = x0 # offsets for UV HWP, QP, B HWP
        angles_c = angles.copy()
        angles_c[0] += a
        angles_c[1] += b
        angles_c[2] += c
        rho_adj = get_Jrho(np.deg2rad(angles_c))
        # rho_adj = adjust_rho(rho_adj, [eta, chi], purity)
        return get_fidelity(rho_adj, rho)

    def loss_func(x0):
        '''Helper function to compute loss between adjusted rho and rho_actual'''
        a,b,c= x0 # offsets for UV HWP, QP, B HWP
        diag_mag_ls = []
        anti_diag_mag_ls = []
        anti_diag_phase_ls = []
        fidelity_ls = []
        purity_ls = []
        # populate lists
        for file in filenames:
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            angles = np.array(angles)
            angles_c = angles.copy()
            angles_c[0] += a
            angles_c[1] += b
            angles_c[2] += c
            # print('before', angles)
            # print('dtype', angles.dtype)
            angles_c =np.deg2rad(np.array(angles_c))
            # print('after', angles_c)
            rho_adj = get_Jrho(angles_c)
            rho_adj = adjust_rho(rho_adj, [eta, chi], purity)
            # rho_adj = la.sqrtm(rho_adj**2 * purity) # normalize

            # take difference of actual and experimental density matrices

            # get magnitude diff
            # rho_actual_mag = np.abs(rho_actual)
            # rho_mag = np.abs(rho)
            # phi_act = np.angle(rho_actual, deg=True)
            # phi = np.angle(rho, deg=True)

            # diff_r = rho_actual_mag / rho_mag
            # diff_phi = phi_act  / phi

            # # log diagonal magntidues and anti-diagonal magnitude
            # # log antidiagonal phase
            # diag_mag_ls.append(diff_r[1,1])
            # anti_diag_mag_ls.append(diff_r[1,2])
            # anti_diag_phase_ls.append(diff_phi[1,2])

            # #### correction ####
            # phi_adj = np.angle(rho_adj, deg=True)
            # rho_adj_mag = np.abs(rho_adj)

            # diff_r_adj = rho_adj_mag / rho_mag
            # diff_phi_adj = phi_adj  / phi

            # diag_mag_c_ls.append(diff_r_adj[1,1])
            # anti_diag_mag_c_ls.append(diff_r_adj[1,2])
            # anti_diag_phase_c_ls.append(diff_phi_adj[1,2])
            diag_mag_ls.append(np.abs(rho[1,1]) - np.abs(rho_adj[1,1]))
            anti_diag_mag_ls.append(np.abs(rho[1,2]) - np.abs(rho_adj[1,2]))
            anti_diag_phase_ls.append(np.angle(rho[1,2]) - np.angle(rho_adj[1,2]))
            fidelity_ls.append(abs(fidelity -get_fidelity(rho_adj, rho)))
            purity_ls.append(abs(purity - get_purity(rho_adj)))

        diag_mag_ls = np.array(diag_mag_ls)
        anti_diag_mag_ls = np.array(anti_diag_mag_ls)
        anti_diag_phase_ls = np.array(anti_diag_phase_ls)
        fidelity_ls = np.array(fidelity_ls)
        purity_ls = np.array(purity_ls)

        return np.sqrt(np.sum(diag_mag_ls**2)) + np.sqrt(np.sum(anti_diag_mag_ls)**2) + np.sqrt(np.sum(anti_diag_phase_ls)**2) + np.sqrt(np.sum(fidelity_ls**2)) + np.sqrt(np.sum(purity_ls**2))

    # optimize
    def get_random_offset():
        '''Helper function to get random offset, in degrees'''
        return np.random.rand(3)*10 - 5 # random offset between -5 and 5 degrees
    
    # get initial random guess
    x0 = get_random_offset()
    best_loss = loss_func(x0)
    grad_offset = x0
    best_offset = x0

    n = 0
    index_since_improvement = 0
    while n < N and abs(best_loss) > loss_lim:
        try:
            # get new x0
            if index_since_improvement == (f*N): # periodic random search (hop)
                x0 = get_random_offset()
                grad_offset = x0
                index_since_improvement = 0
                print('Random search...')
            else:
                gradient = approx_fprime(grad_offset, loss_func, epsilon=1e-8) # epsilon is step size in finite difference
                # if verbose: print(gradient)
                # update angles
                try:
                    x0 = [best_offset[i] - zeta*gradient[i] for i in range(len(best_offset))]
                except TypeError:
                    x0 = best_offset - zeta*gradient
                grad_offset = x0

            # minimize angles
            try:
                soln = minimize(loss_func, x0, bounds=[(-5, 5), (-5, 5), (-5, 5)])
            except ValueError:
                soln = minimize(loss_func, x0, bounds=[(-5, 5)])

            # update best loss and best x0
            x = soln.x
            loss = soln.fun
            if abs(loss) < abs(best_loss):
                best_loss = loss
                best_offset = x0
                index_since_improvement = 0
            else:
                index_since_improvement += 1
            n += 1
            print(f'Iteration {n}: loss = {loss}, best loss = {best_loss}')
        except KeyboardInterrupt:
            break

    print('Best loss: ', best_loss)
    print('Best offset: ', best_offset)
            
def det_noise(filenames, model, UV_HWP_offset, N=100, zeta=.7, f=.1, fidelity_lim = 0.999, do_richard=False):
    '''Determine the probabilties in noise model that minimize the loss function (sum of squares of fidelity differences)
    --
        model: which noise model to assume.
            4 for random r_hh, r_hv, etc
            3 for no cohernece
            1 for coherence
            (i.e., number of fit params)
    
    '''
    print('do richard states', do_richard)

    def loss_func(x0, rho_actual, rho, purity, eta, chi):
        '''Helper function to compute loss between adjusted rho and rho'''    
        # normalize
        if len(x0)>1:
            x0 = x0/np.sum(x0)
        # get fidelity loss
        try:
            fidelity_c, purity_c = get_corrected_fidelity(x0, rho_actual, rho, purity, eta, chi)

            loss =  1 / np.sqrt(fidelity_c)  + abs(purity_c - purity)  # want to maximize fidelity and minimize purity difference and need normalized
        except:
            loss = 1e10
        # fidelity_c, purity_c = get_corrected_fidelity(x0, rho_actual, rho, purity, eta, chi)
        # loss = 1-np.sqrt(fidelity_c)
        return loss

    def minimize_loss(x0, rho_actual, rho, purity, eta, chi, model):
        # normalize guess
        # x0 = x0/np.sum(x0)
        # if not(model==3):
        #     S = minimize(loss_func, x0, bounds = [(-1, 1) for _ in range(model)], args=(rho_actual, rho, purity, eta, chi, ))
        # else:
        S = minimize(loss_func, x0, bounds = [(0, 1) for _ in range(model)], args=(rho_actual, rho, purity, eta, chi))

        fidelity_c, purity_c = get_corrected_fidelity(S.x, rho_actual, rho, purity, eta, chi)
        return  S.x, S.fun, fidelity_c, purity_c # return best prob, best loss, fidelity, purity
    def random_guess(model):
        '''Stick random simplex'''
        # get random simplex
        def do():
            if model >1:
                if not(model==3):
                    rand = 2*np.random.rand(model-1)
                else:
                    rand = np.random.rand(model-1)
                rand = np.sort(rand)
                guess = np.zeros(model)
                guess[0] = rand[0]
                for i in range(1,len(rand)):
                    guess[i] = rand[i] - rand[i-1]
                guess[-1] = 1 - np.sum(guess[:-1])
                # # add negatives
                # if not(model==3):
                #     for i in range(len(guess)):
                #         if np.random.rand()>=0.5:
                #             guess[i]*=-1
                return guess
            else:
                return np.random.rand(1)
        guess = do()
        # while not(is_valid_rho(adjust_E0_rho_general(guess, rho_actual, purity, eta, chi))):
        #     guess = do()
        return guess
    # define functions for loss
    def get_corrected_fidelity(x0, rho_actual, rho, purity, eta, chi):
        '''Adjust the theoretical rho to match the experimental rho, then compute fidelity and purity'''

        rho_c = adjust_E0_rho_general(x0, rho_actual, purity, eta, chi)
        # try:
        # print(rho_c)
        fidelity_c = get_fidelity(rho_c, rho)
        # except ValueError:
        #     fidelity_c = 1e-3
        purity_c = get_purity(rho_c)
        return fidelity_c, purity_c

    if model==4:
        # initialize df to store results
        # results = pd.DataFrame(columns=['eta', 'chi', 'fidelity', 'fidelity_gdcorr', 'purity', 'purity_gdcorr', 'r_hh', 'r_hv', 'r_vh', 'r_vv'])
        results = pd.DataFrame(columns=['eta', 'chi', 'fidelity', 'fidelity_gdcorr', 'purity', 'purity_gdcorr', 'r_hv', 'r_vh', 'r_hh_vv', 'r_vv_hh'])

    elif model==3:
        # initialize df to store results
        results = pd.DataFrame(columns=['eta', 'chi', 'fidelity', 'fidelity_gdcorr', 'purity', 'purity_gdcorr', 'e1', 'e2'])
    elif model==1:
        # initialize df to store results
        results = pd.DataFrame(columns=['eta', 'chi', 'fidelity', 'fidelity_gdcorr', 'purity', 'purity_gdcorr', 'e'])
    elif model==16:
        columns= ['eta', 'chi', 'fidelity', 'fidelity_gdcorr', 'purity', 'purity_gdcorr']
        for l in list('ixyz'):
            for r in list('ixyz'):
                columns.append(f'r_{l}{r}')
        results = pd.DataFrame(columns = columns)

    random_guess = partial(random_guess, model=model)
    for file in tqdm(filenames):
        print('----------')
        print(file)

        try:
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
        except:
            trial, rho, unc, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
            eta, chi = None, None
        
        # adjust UV_HWP
        angles[0]+=UV_HWP_offset
        rho_actual = get_Jrho(angles=np.deg2rad(angles))
        fidelity = get_fidelity(rho_actual, rho)

        # perform loss minimization
        # get initial random guess
        print('original fidelity', fidelity)
        x0 = random_guess()
        print('first guess', x0)
        ploss_func = lambda x: loss_func(x, rho_actual, rho, purity, eta, chi)
        best_loss = ploss_func(x0)
        best_prob = x0
        grad_prob = x0
        try:
            fidelity_c, purity_c = get_corrected_fidelity(x0, rho_actual, rho, purity, eta, chi)
        except:
            fidelity_c, purity_c = 1e-3, 1e-3
        print('first try fidelity_c', fidelity_c)
        best_fidelity = fidelity_c
        if best_fidelity > 1:
            best_fidelity = 1e-3
        #     if np.round(best_fidelity, 3) == 1.0:
        #         best_fidelity = 1
        #     else:
        #         best_fidelity = 1e-3
        best_purity = purity_c
        n = 0
        index_since_improvement = 0
        while n < N and best_fidelity < fidelity_lim:
            print(f'Index {n}: Fidelity = {fidelity_c}. Best fidelity = {best_fidelity}')
            print('best guess', best_prob)
            # print('Guess', x0)

            if index_since_improvement % (f*N)==0: # periodic random search (hop)
                x0 = random_guess()
                index_since_improvement = 0
                print('Random search...')
            else:
                gradient = approx_fprime(grad_prob, ploss_func, epsilon=1e-8) # epsilon is step size in finite difference
                # if verbose: print(gradient)
                # update angles
                x0 = [best_prob[i] - zeta*gradient[i] for i in range(len(best_prob))]
                grad_prob = x0
            # minimize loss
            try:
                x, loss, fidelity_c, purity_c = minimize_loss(x0, rho_actual, rho, purity, eta, chi, model)
            except ValueError:
                x0 = random_guess()
                x, loss, fidelity_c, purity_c = minimize_loss(x0, rho_actual, rho, purity, eta, chi, model)

            if np.round(fidelity_c, 3) == 1.0:
                fidelity_c = 1
                best_fidelity = fidelity_c
                best_purity = purity_c
                best_loss = loss
                best_prob = x
              
            # update best loss and best x0
            if fidelity_c > best_fidelity and fidelity_c <=1 and purity_c < 1:
                best_fidelity = fidelity_c
                best_purity = purity_c
                best_loss = loss
                best_prob = x
                index_since_improvement = 0
            else:
                index_since_improvement += 1
            
            if 0<fidelity_c <= 1:
                n += 1
            else:
                x0 = random_guess()
            # except ValueError:
            #     print('problem')
            #     x0 = random_guess()
        # best_prob /= np.sum(best_prob) # normalize guess, since that's what we input into loss
        print('Best fidelity: ', best_fidelity)
        print('Best fidelity this func', get_corrected_fidelity(best_prob, rho_actual, rho, purity, eta, chi)[0])
        print('Best fidelity rechecked rho methods', get_fidelity(adjust_E0_rho_general(best_prob, rho_actual, purity, eta, chi), rho))
        print('Best prob: ', best_prob)
        print('Best loss: ', best_loss)
        print('n', n)

        # if model == 4:
        #     results = pd.concat([results, pd.DataFrame.from_records([{'eta': eta, 'chi': chi, 'fidelity': fidelity, 'fidelity_gdcorr': best_fidelity, 'purity': purity, 'purity_gdcorr': best_purity, 'r_hh': best_prob[0], 'r_hv': best_prob[1], 'r_vh': best_prob[2], 'r_vv': best_prob[3]}])])
        if model == 4:
            results = pd.concat([results, pd.DataFrame.from_records([{'eta': eta, 'chi': chi, 'fidelity': fidelity, 'fidelity_gdcorr': best_fidelity, 'purity': purity, 'purity_gdcorr': best_purity,  'r_hv': best_prob[0], 'r_vh': best_prob[1], 'r_hh_vv': best_prob[2],'r_vv_hh': best_prob[3]}])])
        elif model ==3:
            results =pd.concat([results, pd.DataFrame.from_records([{'eta': eta, 'chi': chi, 'fidelity': fidelity, 'fidelity_gdcorr': best_fidelity, 'purity': purity, 'purity_gdcorr': best_purity, 'e1': best_prob[0], 'e2': best_prob[1]}])]) 
        elif model==1:
            results =pd.concat([results, pd.DataFrame.from_records([{'eta': eta, 'chi': chi, 'fidelity': fidelity, 'fidelity_gdcorr': best_fidelity, 'purity': purity, 'purity_gdcorr': best_purity, 'e': best_prob[0],}])]) 
        elif model==16:
            soln_dict = {'eta': eta, 'chi': chi, 'fidelity': fidelity, 'fidelity_gdcorr': best_fidelity, 'purity': purity, 'purity_gdcorr': best_purity}
            best_prob = best_prob.reshape((4,4))
            for i, l in enumerate(list('ixyz')):
                for j, r in enumerate(list('ixyz')):
                    soln_dict[f'r_{l}{r}'] = best_prob[i, j]
            results = pd.concat([results, pd.DataFrame.from_records([soln_dict])])

    # save results
    print('saving!')
    if not(isdir(join(DATA_PATH, f'noise_{UV_HWP_offset}'))):
        os.makedirs(join(DATA_PATH, f'noise_{UV_HWP_offset}'))
    if not(do_richard):
        results.to_csv(join(DATA_PATH, f'noise_{UV_HWP_offset}/noise_model_{model}.csv'))
    else:
        results.to_csv(join(DATA_PATH, f'noise_{UV_HWP_offset}/noise_model_{model}_richard.csv'))

def det_noise_s(filenames, model, UV_HWP_offset, N=250, zeta=1, f=.1, loss_lim = 1e-2):
    '''Determine the probabilties in noise model that minimize the loss function (sum of squares of stokes parameter differences)
    --
        model: which noise model to assume.
            4 for random r_hh, r_hv, etc
            3 for no cohernece
            1 for coherence
            (i.e., number of fit params)
    
    '''

    def loss_func(x0, rho_actual, purity, eta, chi, S_targ):
        '''Helper function to compute loss between adjusted rho and rho'''    
        # normalize
        if len(x0)>1:
            x0 = x0/np.sum(x0)
        # get fidelity loss
        # get rho adj
        rho_adj = adjust_E0_rho_general(x0, rho_actual, purity, eta, chi)
        # get stokes params
        St = get_expec_vals(rho_adj)
        # loss is frobenius norm of difference between stokes params
        loss = abs(1 / np.sqrt(abs(np.trace(np.sqrt(np.sqrt(S_targ) @ St @ np.sqrt(S_targ))))))
        # print(loss)
        return loss

    def minimize_loss(x0, rho_actual, purity, eta, chi,model, S_targ):
        # normalize guess
        # x0 = x0/np.sum(x0)
        if not(model==3 or model==1):
            S = minimize(loss_func, x0, bounds = [(-1, 1) for _ in range(model)], args=(rho_actual, purity, eta, chi, S_targ))
        else:
            S = minimize(loss_func, x0, bounds = [(0, 1) for _ in range(model)], args=(rho_actual, purity, eta, chi, S_targ))
        x= S.x
        # get adjusted rho
        rho_c = adjust_E0_rho_general(x, rho_actual, purity, eta, chi)
        S_c = get_expec_vals(rho_c)
        return  x, S.fun, S_c
    def random_guess(model):
        '''Stick random simplex'''
        def do():
            # get random simplex
            if model >1:
                if not(model==3):
                    rand = 2*np.random.rand(model-1)
                else:
                    rand = np.random.rand(model-1)
                rand = np.sort(rand)
                guess = np.zeros(model)
                guess[0] = rand[0]
                for i in range(1,len(rand)):
                    guess[i] = rand[i] - rand[i-1]
                guess[-1] = 1 - np.sum(guess[:-1])
                # add negatives
                # if not(model==3):
                #     for i in range(len(guess)):
                #         if np.random.rand()>=0.5:
                #             guess[i]*=-1
            else:
                guess =  np.random.rand(1)
            return guess
        guess = do()
        # while not(is_valid_rho(adjust_E0_rho_general(guess, rho_actual, purity, eta, chi))):
        #     guess = do()
        return guess

    def get_corrected_fidelity(x0, rho_actual, rho, purity, eta, chi):
        '''Adjust the theoretical rho to match the experimental rho, then compute fidelity and purity'''

        rho_c = adjust_E0_rho_general(x0, rho_actual, purity, eta, chi)
        
        fidelity_c = get_fidelity(rho_c, rho)
        
        purity_c = get_purity(rho_c)
        return fidelity_c, purity_c

    if model==4:
        # initialize df to store results
        results = pd.DataFrame(columns=['eta', 'chi', 'fidelity', 'fidelity_c', 'purity', 'purity_c', 'r_hh', 'r_hv', 'r_vh', 'r_vv'])

    elif model==3:
        # initialize df to store results
        results = pd.DataFrame(columns=['eta', 'chi', 'fidelity', 'fidelity_c', 'purity', 'purity_c', 'e1', 'e2'])
    elif model==1:
        # initialize df to store results
        results = pd.DataFrame(columns=['eta', 'chi', 'fidelity', 'fidelity_c', 'purity', 'purity_c', 'e'])
    elif model==16:
        columns= ['eta', 'chi', 'fidelity', 'fidelity_c', 'purity', 'purity_c']         
        for l in list('ixyz'):
            for r in list('ixyz'):
                columns.append(f'r_{l}{r}')
        results = pd.DataFrame(columns = columns)

    random_guess = partial(random_guess, model=model)

    for file in tqdm(filenames):
        print('----------')
        print(file)
        try:
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
        except:
            trial, rho, unc, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
            eta, chi = None, None
        
        # get experimental stokes params, which are the target
        S_targ = get_expec_vals(rho)

        # adjust UV_HWP
        angles[0]+=UV_HWP_offset
        rho_actual = get_Jrho(angles=np.deg2rad(angles))
        fidelity = get_fidelity(rho_actual, rho)

        # perform loss minimization
        # get initial random guess
        print('original fidelity', fidelity)
        print('original diff', loss_func([0,0,0,0], rho_actual, purity, eta, chi, S_targ))
        x0 = random_guess()
        print('first guess', x0)
        ploss_func = lambda x: loss_func(x, rho_actual, purity, eta, chi, S_targ)
        loss_c = ploss_func(x0)
        best_loss = loss_c
        best_prob = x0
        best_S = get_expec_vals(adjust_E0_rho_general(x0, rho_actual, purity, eta, chi))
        fidelity_c, purity_c = get_corrected_fidelity(x0, rho_actual, rho, purity, eta, chi)
        best_fidelity = fidelity_c
        best_purity = purity_c
        n=0
        index_since_improvement = 0
        while n < N and best_loss > loss_lim:
            print(f'Index {n}: Loss = {loss_c}. Best loss = {best_loss}')
            print('fidelity', fidelity_c)

            
            # if index_since_improvement % (f*N)==0: # periodic random search (hop)
            #     x0 = random_guess()
            #     index_since_improvement = 0
            #     print('Random search...')
            # else:
            #     gradient = approx_fprime(grad_prob, ploss_func, epsilon=1e-8) # epsilon is step size in finite difference
            #     print(gradient)
            #     # update angles
            #     x0 = [best_prob[i] - zeta*gradient[i] for i in range(len(best_prob))]
            #     grad_prob = x0
            # minimize loss
            # try:
            x0 = random_guess()
            x, loss, S_c = minimize_loss(x0, rho_actual, purity, eta, chi, model, S_targ)
            fidelity_c, purity_c = get_corrected_fidelity(x, rho_actual, rho, purity, eta, chi)
            # update best loss and best x0
            if loss < best_loss:
                best_fidelity = fidelity_c
                best_purity = purity_c
                best_loss = loss
                best_prob = x
                best_S = S_c
                index_since_improvement = 0
            else:
                index_since_improvement += 1
            
            n += 1
          
           
        print('Best prob: ', best_prob)
        print('Best loss: ', best_loss)
        print('n', n)
        print('S targ', S_targ)
        print('S best', best_S)

        if model == 4:
            results = pd.concat([results, pd.DataFrame.from_records([{'eta': eta, 'chi': chi, 'fidelity':fidelity, 'fidelity_c':best_fidelity, 'purity':purity, 'best_purity':best_purity, 'r_hh': best_prob[0], 'r_hv': best_prob[1], 'r_vh': best_prob[2], 'r_vv': best_prob[3]}])])
        elif model ==3:
            results =pd.concat([results, pd.DataFrame.from_records([{'eta': eta, 'chi': chi, 'fidelity':fidelity, 'fidelity_c':best_fidelity, 'purity':purity, 'best_purity':best_purity, 'e1': best_prob[0], 'e2': best_prob[1]}])]) 
        elif model==1:
            results =pd.concat([results, pd.DataFrame.from_records([{'eta': eta, 'chi': chi, 'fidelity':fidelity, 'fidelity_c':best_fidelity, 'purity':purity, 'best_purity':best_purity, 'e': best_prob[0],}])]) 
        elif model==16:
            soln_dict = {'eta': eta, 'chi': chi, 'fidelity':fidelity, 'fidelity_c':best_fidelity, 'purity':purity, 'best_purity':best_purity}
            best_prob = best_prob.reshape((4,4))
            for i, l in enumerate(list('ixyz')):
                for j, r in enumerate(list('ixyz')):
                    soln_dict[f'r_{l}{r}'] = best_prob[i, j]
            results = pd.concat([results, pd.DataFrame.from_records([soln_dict])])

    # save results
    print('saving!')
    if not(isdir(join(DATA_PATH, f'noise_{UV_HWP_offset}_W'))):
        os.makedirs(join(DATA_PATH, f'noise_{UV_HWP_offset}_W'))
    results.to_csv(join(DATA_PATH, f'noise_{UV_HWP_offset}_W/noise_model_{model}.csv'))

def det_stokes_ratios(files_45, files_30, files_60, model=1, UV_HWP_offset=1.029):
    fig, ax = plt.subplots(1, 3, figsize=(20,7))
    eta_ls = [45, 30, 60]
    tab20_cmap = plt.get_cmap('tab20')
    for i, files in enumerate([files_45, files_30, files_60]):
        # get 15 stoke ratios
        S1 = []
        S2 = []
        S3 = []
        S4 = []
        S5 = []
        S6 = []
        S7 = []
        S8 = []
        S9 = []
        S10 = []
        S11 = []
        S12 = []
        S13 = []
        S14 = []
        S15 = []
        chi_ls = []
        for file in files:
            # read in rho
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            # get adjusted rho
            angles[0]+=UV_HWP_offset
            rho_actual = get_Jrho(angles=np.deg2rad(angles))
            rho_adj = load_saved_get_E0_rho_c(rho_actual, [eta, chi], purity, model, do_W = False, UV_HWP_offset = UV_HWP_offset)
            # get stokes params
            S = get_expec_vals(rho)
            S_adj = get_expec_vals(rho_adj)
            # get stokes ratios; try real part or make 0
            S_r_real =np.real(S - S_adj)
            S_r_real = S_r_real.reshape((16,1))

            S1.append(S_r_real[1])
            S2.append(S_r_real[2])
            S3.append(S_r_real[3])
            S4.append(S_r_real[4])
            S5.append(S_r_real[5])
            S6.append(S_r_real[6])
            S7.append(S_r_real[7])
            S8.append(S_r_real[8])
            S9.append(S_r_real[9])
            S10.append(S_r_real[10])
            S11.append(S_r_real[11])
            S12.append(S_r_real[12])
            S13.append(S_r_real[13])
            S14.append(S_r_real[14])
            S15.append(S_r_real[15])
  
            # get chi
            chi_ls.append(chi)
        # plot
        ax[i].scatter(chi_ls, S1, label='$\sigma_I \otimes \sigma_X $', color=tab20_cmap(0))
        ax[i].scatter(chi_ls, S2, label='$\sigma_I \otimes \sigma_Y $', color=tab20_cmap(1))
        ax[i].scatter(chi_ls, S3, label='$\sigma_I \otimes \sigma_Z $', color=tab20_cmap(2))
        # ax[i].scatter(chi_ls, S4, label='$\sigma_X \otimes \sigma_I $', color=tab20_cmap(3))
        ax[i].scatter(chi_ls, S5, label='$\sigma_X \otimes \sigma_X $', color=tab20_cmap(4))
        ax[i].scatter(chi_ls, S6, label='$\sigma_X \otimes \sigma_Y $', color=tab20_cmap(5))
        ax[i].scatter(chi_ls, S7, label='$\sigma_X \otimes \sigma_Z $', color=tab20_cmap(6))
        # ax[i].scatter(chi_ls, S8, label='$\sigma_Y\otimes \sigma_I $', color=tab20_cmap(7))
        # ax[i].scatter(chi_ls, S9, label='$\sigma_Y\otimes \sigma_X $', color=tab20_cmap(8))
        ax[i].scatter(chi_ls, S10, label='$\sigma_Y\otimes \sigma_Y $', color=tab20_cmap(9))
        ax[i].scatter(chi_ls, S11, label='$\sigma_Y\otimes \sigma_Z $', color=tab20_cmap(10))
        # ax[i].scatter(chi_ls, S12, label='$\sigma_Z\otimes \sigma_I $', color=tab20_cmap(11))
        # ax[i].scatter(chi_ls, S13, label='$\sigma_Z\otimes \sigma_X $', color=tab20_cmap(12))
        # ax[i].scatter(chi_ls, S14, label='$\sigma_Z\otimes \sigma_Y $', color=tab20_cmap(13))
        ax[i].scatter(chi_ls, S15, label='$\sigma_Z\otimes \sigma_Z $', color=tab20_cmap(14))
        ax[i].set_title(f'$\eta = {eta_ls[i]}$')
        ax[i].set_xlabel('$\chi$')

        # fit 5th order polynomial
        p1 = np.polyfit(chi_ls, S1, 5)
        p2 = np.polyfit(chi_ls, S2, 5)
        p3 = np.polyfit(chi_ls, S3, 5)
        p4 = np.polyfit(chi_ls, S4, 5)
        p5 = np.polyfit(chi_ls, S5, 5)
        p6 = np.polyfit(chi_ls, S6, 5)
        p7 = np.polyfit(chi_ls, S7, 5)
        p8 = np.polyfit(chi_ls, S8, 5)
        p9 = np.polyfit(chi_ls, S9, 5)
        p10 = np.polyfit(chi_ls, S10, 5)
        p11 = np.polyfit(chi_ls, S11, 5)
        p12 = np.polyfit(chi_ls, S12, 5)
        p13 = np.polyfit(chi_ls, S13, 5)
        p14 = np.polyfit(chi_ls, S14, 5)
        p15 = np.polyfit(chi_ls, S15, 5)

        # plot
        chi_lin = np.linspace(min(chi_ls), max(chi_ls), 100)
        ax[i].plot(chi_lin, np.polyval(p1, chi_lin), linestyle='--', color=tab20_cmap(0))
        ax[i].plot(chi_lin, np.polyval(p2, chi_lin), linestyle='--', color=tab20_cmap(1))
        ax[i].plot(chi_lin, np.polyval(p3, chi_lin), linestyle='--', color=tab20_cmap(2))
        # ax[i].plot(chi_lin, np.polyval(p4, chi_lin), linestyle='--', color=tab20_cmap(3))
        ax[i].plot(chi_lin, np.polyval(p5, chi_lin), linestyle='--', color=tab20_cmap(4))
        ax[i].plot(chi_lin, np.polyval(p6, chi_lin), linestyle='--', color=tab20_cmap(5))
        ax[i].plot(chi_lin, np.polyval(p7, chi_lin), linestyle='--', color=tab20_cmap(6))
        # ax[i].plot(chi_lin, np.polyval(p8, chi_lin), linestyle='--', color=tab20_cmap(7))
        # ax[i].plot(chi_lin, np.polyval(p9, chi_lin), linestyle='--', color=tab20_cmap(8))
        ax[i].plot(chi_lin, np.polyval(p10, chi_lin), linestyle='--', color=tab20_cmap(9))
        ax[i].plot(chi_lin, np.polyval(p11, chi_lin), linestyle='--', color=tab20_cmap(10))
        # ax[i].plot(chi_lin, np.polyval(p12, chi_lin), linestyle='--', color=tab20_cmap(11))
        # ax[i].plot(chi_lin, np.polyval(p13, chi_lin), linestyle='--', color=tab20_cmap(12))
        # ax[i].plot(chi_lin, np.polyval(p14, chi_lin), linestyle='--', color=tab20_cmap(13))
        ax[i].plot(chi_lin, np.polyval(p15, chi_lin), linestyle='--', color=tab20_cmap(14))

    if model is None:
        model == 'Base'

    ax[0].legend(ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Stokes differences, $S_{expt} - S_{adj}$'+f', Model {model}')
    plt.savefig(join(DATA_PATH, f'stokes_diff_{model}_{UV_HWP_offset}.pdf'))

def plot_adj():
    fig, ax = plt.subplots(2, 3, figsize=(20,7))
    for j, model in enumerate([1, 3, 4]):

        adjust_df = pd.read_csv(f'../../framework/decomp_test/noise_model_{model}.csv')

        for i, eta in enumerate(list(set(adjust_df['eta'].values))):
            eta_adjust_df = adjust_df.loc[np.round(adjust_df['eta'], 4) == np.round(eta, 4)]
            # get chi_ls
            chi_ls = eta_adjust_df['chi'].values
            fidelity_c_ls = eta_adjust_df['fidelity_gdcorr'].values
            fidelity_ls = eta_adjust_df['fidelity'].values
            purity_c_ls = eta_adjust_df['purity_gdcorr'].values
            purity_ls = eta_adjust_df['purity'].values

            ax[0, i].scatter(chi_ls, fidelity_c_ls, label=f'model {model}')
            ax[1, i].scatter(chi_ls, purity_c_ls, label=f'model {model}')
            if j == 2:
                ax[0, i].plot(chi_ls, fidelity_ls, linestyle='--', color='r', label='actual')
                ax[1, i].plot(chi_ls, purity_ls, linestyle='--', color='r', label='actual')

            ax[0, i].legend()
            ax[1, i].legend()
            
            ax[0, i].set_ylabel('Fidelity')
            ax[1, i].set_ylabel('Purity')

            ax[1, i].set_xlabel('$\chi$')
            ax[0, i].set_title('$\eta = %.3g$'%eta)
    plt.tight_layout()
    plt.suptitle('Adjustment for noise')
    plt.subplots_adjust(top=0.9)
    plt.savefig(join(DATA_PATH, 'noise_plots.pdf'))
            
if __name__ == '__main__':
    # set filenames for computing W values
    ## new names ##

    ## old ##
    # filenames_45 = ["rho_('E0', (45.0, 0.0))_20.npy", "rho_('E0', (45.0, 18.0))_20.npy", "rho_('E0', (45.0, 36.0))_20.npy", "rho_('E0', (45.0, 54.0))_20.npy", "rho_('E0', (45.0, 72.0))_20.npy", "rho_('E0', (45.0, 90.0))_20.npy"]
    # filenames_60= ["rho_('E0', (59.99999999999999, 0.0))_22.npy", "rho_('E0', (59.99999999999999, 18.0))_22.npy", "rho_('E0', (59.99999999999999, 36.0))_22.npy", "rho_('E0', (59.99999999999999, 54.0))_22.npy", "rho_('E0', (59.99999999999999, 72.0))_22.npy", "rho_('E0', (59.99999999999999, 90.0))_22.npy"]

    # filenames = filenames_45 + filenames_60

    # settings_45 = [[45.0,13.107759739471968,45.0], [40.325617881787,32.45243475604995,45.0], [35.319692011068646,32.80847131578413,45.0], [29.99386625322187,32.59712114540248,45.0], [26.353505137451158,32.91656908476468,44.71253931908844], [20.765759133476752,32.763298596034836,45.0]]
    # settings_60 = [[36.80717351236577,38.298986094951985,45.0], [35.64037134135345,36.377936778443754,44.99999], [32.421520781235735,35.46619180422062,44.99998], [28.842682522467676,34.97796909446873,44.61235], [25.8177216842833,34.72228985431089,44.74163766], [21.614459228879422,34.622127766985436,44.9666]]
    # settings = settings_45 + settings_60
    # analyze rho files
    # id = 'richard'

    ## do calculations ##

    
    models = [1, 3]
    do_W = False
    do_richard = False

    if not(do_richard):
        # set path
        current_path = dirname(abspath(__file__))
        DATA_PATH = join(current_path, '../../framework/decomp_test/')
        UV_HWP_offset = 1.029 # from Alec's recallibration
        filenames_30= ["rho_('E0', (29.999999999999996, 0.0))_34.npy", "rho_('E0', (29.999999999999996, 18.0))_34.npy", "rho_('E0', (29.999999999999996, 36.0))_34.npy", "rho_('E0', (29.999999999999996, 54.0))_34.npy", "rho_('E0', (29.999999999999996, 72.0))_34.npy", "rho_('E0', (29.999999999999996, 90.0))_34.npy"]
        filenames_45 = ["rho_('E0', (45.0, 0.0))_33.npy", "rho_('E0', (45.0, 18.0))_33.npy", "rho_('E0', (45.0, 36.0))_33.npy", "rho_('E0', (45.0, 54.0))_33.npy", "rho_('E0', (45.0, 72.0))_33.npy", "rho_('E0', (45.0, 90.0))_33.npy"]
        filenames_60 = ["rho_('E0', (59.99999999999999, 0.0))_32.npy", "rho_('E0', (59.99999999999999, 18.0))_32.npy", "rho_('E0', (59.99999999999999, 36.0))_32.npy", "rho_('E0', (59.99999999999999, 54.0))_32.npy", "rho_('E0', (59.99999999999999, 72.0))_32.npy", "rho_('E0', (59.99999999999999, 90.0))_32.npy"]
        filenames = filenames_45 + filenames_30 + filenames_60

    else:
        DATA_PATH= 'richard'
        UV_HWP_offset = 0 # from Alec's recallibration
        alphas = [np.pi/6, np.pi/4]
        betas = np.linspace(0.001, np.pi/2, 6)
        states_names = []
        states = []

        for alpha in alphas:
            for beta in betas:
                states_names.append((np.rad2deg(alpha), np.rad2deg(beta)))
                states.append((alpha, beta))

        filenames = []

        for i, state_n in enumerate(states_names):
            filenames.append(f"rho_('E0', {state_n})_26.npy")

    # for model in models:
    #     comp_w_adj(filenames, model=model)

        # if not(do_W):
        #     det_noise(filenames, model, UV_HWP_offset, do_richard=do_richard)
        # else:
        #     det_noise_s(filenames, model, UV_HWP_offset)
    model=3
    id = f'{UV_HWP_offset}_fixed_angles_{do_richard}'
    analyze_rhos(filenames,id=id, UV_HWP_offset = UV_HWP_offset, do_W=do_W, model=model) # calculate csv with witness vals
    if not(do_W):
        make_plots_E0(f'rho_analysis_{id}.csv', paper=True) # make plots based on witness calcs
    else:
        make_plots_E0(f'rho_analysis_{id}_W.csv')
    # for model in [None, 0, 1, 3]:
    #     det_stokes_ratios(filenames_45, filenames_30, filenames_60, model=model)
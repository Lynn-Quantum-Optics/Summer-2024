'''
AuthorL Lev Gruber
Last Update: 5/17/2024

This file uses code from Oscar's process_expt_richard.py in Summer 2023 to brute force
calculate eta and chi values (for the state cos(eta)PHI+ + sin(eta)e^i*chi PHI-) in which W is positive
and W' is negative. 

It does so by:
1. Creating theoretical density matrices at a variety of eta values and 1-2 chi values
2. Computing both W and W' witnesses for each density matrix
3. Pushing values with +W and -W' into one df, and values with +W, +W' into another. 

In the future, I hope to add functionality to compute V witnesses for states for which W and W' are positive.
'''

# Imports
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from uncertainties import ufloat
from uncertainties import unumpy as unp

    # Original code from Oscar Scholin in Summer 2023
from sample_rho import *
from rho_methods import *

# Helper methods 
def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho_psi(eta, chi):
    """
    Calculates the density matrix (rho) for a given set of parameters (eta, chi) in Psi.

    Parameters:
    eta (float): The parameter eta.
    chi (float): The parameter chi.

    Returns:
    numpy.ndarray: The density matrix (rho) calculated based on the given parameters.
    """
    H = ket([1,0])
    V = ket([0,1])
    
    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)

    psi = np.cos(eta)*PSI_PLUS + np.exp(1j*chi)*np.sin(eta)*PSI_MINUS

    rho = psi @ psi.conj().T

    return rho

def get_theo_rho_phi(eta, chi):
    """
    Calculates the density matrix (rho) for a given set of parameters (eta, chi) in Phi.

    Parameters:
    eta (float): The parameter eta.
    chi (float): The parameter chi.

    Returns:
    numpy.ndarray: The density matrix (rho) calculated based on the given parameters.
    """
    H = ket([1,0])
    V = ket([0,1])
    
    PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)

    phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PHI_MINUS

    rho = phi @ phi.conj().T

    return rho


def analyze_rhos(rho_actuals, id='id'):
    '''; 
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

    for i, rho_actual in enumerate(rho_actuals)

        # calculate W and W' theory
        W_T_ls = compute_witnesses(rho = rho_actual) # theory
        W_AT_ls = compute_witnesses(rho = rho_actual, expt_purity=purity, angles=[eta, chi]) # adjusted theory

        # calculate W and W' expt
        W_expt_ls = compute_witnesses(rho = rho, expt=True, counts=unp.uarray(un_proj, un_proj_unc))

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

        if eta is not None and chi is not None:
            adj_fidelity= get_fidelity(adjust_rho(rho_actual, [eta, chi], purity), rho)

            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'eta':eta, 'chi':chi, 'fidelity':fidelity, 'purity':purity, 'AT_fidelity':adj_fidelity,
            'W_min_T': W_min_T, 'Wp_t1_T':Wp_t1_T, 'Wp_t2_T':Wp_t2_T, 'Wp_t3_T':Wp_t3_T,'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

        else:
            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'fidelity':fidelity, 'purity':purity, 'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

    # save df
    print('saving!')
    df.to_csv(join(DATA_PATH, f'rho_analysis_{id}.csv'))

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

        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
        ax.errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue')


        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
        ax.errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon')

        ax.set_title(f'$\eta = {np.round(60,3)}\degree$', fontsize=33)
        ax.set_ylabel('Witness value', fontsize=31)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.legend(ncol=2, fontsize=25)
        ax.set_xlabel('$\chi$', fontsize=31)
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

            ax[i].set_title(f'$\eta = {np.round(eta,3)}\degree$', fontsize=33)
            ax[i].set_ylabel('Witness value', fontsize=31)
            ax[i].tick_params(axis='both', which='major', labelsize=25)
            ax[i].legend(ncol=2, fontsize=25)
            ax[i].set_xlabel('$\chi$', fontsize=31)
            # ax[1,i].set_ylabel('Value', fontsize=31)
            # ax[1,i].legend()

            
    plt.suptitle('Witnesses for $E_0$ states, $\cos(\eta)|\Phi^+\\rangle + \sin(\eta)e^{i \chi}|\Phi^-\\rangle $', fontsize=35)
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'exp_witnesses_E0_{id}.pdf'))
    plt.show()

if __name__ == '__main__':
    # set filenames for computing W values
    
    etas = [np.pi/3]
    chis = np.linspace(0.001, np.pi/2, 6)
    states_names = []
    states = []

    for eta in etas:
        for chi in chis:
            states_names.append((np.rad2deg(eta), np.rad2deg(chi)))
            states.append((eta, chi)) 
    rho_actuals = []
    for i, state_n in enumerate(states_names):
        rad_angles = states[i]
        rho_actuals.append(get_theo_rho_phi(rad_angles[0],rad_angles[1]))

    # analyze rho files
    id = 'rho_5162024'
    analyze_rhos(rho_actuals, id=id)
    make_plots_E0(f'rho_analysis_{id}.csv')




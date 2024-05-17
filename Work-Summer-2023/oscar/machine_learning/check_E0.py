# file to check witness values for E1 state
from rho_methods import *
from sample_rho import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_W_Wps(gen_funcs, eta_ls, chi_ls):
    '''Find the W and Wp values for E0_p states for a range of eta and chi values.
    
    Parameters:
    gen_func: function that generates the state
    eta_ls: list of eta values to check, in degrees
    chi_min: minimum chi value to check, in degrees
    chi_max: maximum chi value to check, in degrees
    num: number of chi values to check between chi_min and chi_max

    Returns:
    W_df: dataframe of W values
    Wp_df: dataframe of Wp values
    W_min, Wp_min: lists of minimum W and Wp values for each eta
    '''
    eta_ls = np.deg2rad(np.array(eta_ls))
    chi_ls = np.deg2rad(chi_ls)
    # hold 6 W values
    W_min_dict = {'eta': eta_ls, 'W_min': []}
    # hold 9 Wp values
    Wp_min_dict = {'eta': eta_ls, 'Wp_min': []}

    fig, ax = plt.subplots(len(eta_ls), len(gen_funcs), figsize=(10, 10), sharex=True)

    for l, gen_func in enumerate(gen_funcs):

        for k, eta in enumerate(eta_ls):
            W_dict = {'W0': [], 'W1': [], 'W2': [], 'W3': [], 'W4': [], 'W5': []}
            Wp_dict = {'Wp0': [], 'Wp1': [], 'Wp2': [], 'Wp3': [], 'Wp4': [], 'Wp5': [], 'Wp6': [], 'Wp7': [], 'Wp8': []}
            W_min_eta = []
            Wp_min_eta = []

            for chi in chi_ls:
                rho = gen_func(eta, chi)
                W_all = compute_witnesses(rho, return_all=True)
                W_min = min(W_all[:6])
                Wp_min = min(W_all[6:])
                for i in range(6):
                    W_dict[f'W{i}'].append(W_all[i])
                for i in range(9):
                    Wp_dict[f'Wp{i}'].append(W_all[i+6])
                W_min_eta.append(W_min)
                Wp_min_eta.append(Wp_min)

            # scatter the min W and Wp values
            # extract name of function that generated the state
            name = gen_func.__name__.split('_')
            if len(name) == 2:
                name_n = name[1]
            elif len(name) == 3:
                name_n = name[1] + '_' + name[2]
            ax[k][l].scatter(np.rad2deg(chi_ls), W_min_eta, label=f'$W$, {name_n}')
            ax[k][l].scatter(np.rad2deg(chi_ls), Wp_min_eta, label=f'$W\'$, {name_n}')
            ax[k][l].set_title(f'$\eta = {np.round(np.rad2deg(eta_ls[k]), 3)}$')
            ax[k][l].legend()
            ax[k][l].set_ylabel('Witness value')

            # determine for each chi what the minimum W and Wp values are
            W_min_dict['W_min'].append(W_min_eta)
            Wp_min_dict['Wp_min'].append(Wp_min_eta)

            # save the dictionaries!
            W_df = pd.DataFrame(W_dict)
            Wp_df = pd.DataFrame(Wp_dict)
        
            # save to csv
            W_df.to_csv(f'W_df_{eta}_{name_n}.csv')
            Wp_df.to_csv(f'Wp_df_{eta}_{name_n}.csv')
        
    # save image
    ax[-1][0].set_xlabel('$\chi$')
    ax[-1][1].set_xlabel('$\chi$')
    plt.tight_layout()

    plt.savefig(f'W_Wp.pdf')
    
    # save the minimum W and Wp values
    # W_min_dict = pd.DataFrame(W_min_dict)
    # Wp_min_dict = pd.DataFrame(Wp_min_dict)
    # W_min_dict.to_csv('W_min.csv')
    # Wp_min_dict.to_csv('Wp_min.csv')
    return W_df, Wp_df, W_min_dict, Wp_min_dict

def do_plot(eta_ls, chi_ls, W_min, Wp_min, name):
    fig, ax = plt.subplots(len(eta_ls), 1, figsize=(10, 10), sharex=True)
    for i in range(len(eta_ls)):
        ax[i].scatter(np.rad2deg(chi_ls), W_min[i], label='W')
        ax[i].scatter(np.rad2deg(chi_ls), Wp_min[i], label='Wp')
        ax[i].set_title(f'eta = {np.rad2deg(eta_ls[i])}')
        ax[i].legend()
        ax[i].set_ylabel('Witness value')
    ax[-1].set_xlabel('$\chi$')
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')


if __name__ == '__main__':
    # plot min W and min Wp values for each eta
    # eta_ls = [30, 45]
    # find_W_Wps(eta_ls, num=20)]
    eta_ls = [30, 45]
    chi_min = 0
    chi_max = 90
    num = 6
    chi_ls = np.linspace(chi_min, chi_max, num)

    find_W_Wps([get_E0_p, get_E0], eta_ls, chi_ls)
    # print(get_E0_p.__name__)

    # get_concurrence(get_E0_p(np.deg2rad(45), np.deg2rad(90)))
    

    





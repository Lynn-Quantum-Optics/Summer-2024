# file to compare the distributions of properties of random density matrices generated from my get_random_hurwitz() code and the code from the group in summer 2022/ fall 2022 / spring 2023

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import scipy.io as sio

from sample_rho import *
from rho_methods import *

# define list of csvs
DATA_PATH = 'random_gen/test'

def comp_info(data_ls, names, savename, titlename, do_witnesses=True):
    ''' Function to plot properties of states and witness values, including the states undetected by the witnesses. 
    
    params:
        data_ls: list of csvs to read in
        names: list of names of data
        savename: name to save figure as
        titlename: title of figure
        do_witnesses: bool, whether to plot witness values
    '''
    if do_witnesses:
        # read in data and calculate quantities
        # have df to store info about states
        info_df = pd.DataFrame({'name':[], 'ent':[], 'undet_W_0':[], 'undet_Wp_0':[], 'undet_all_0':[], 'w_cond_ent':[], 'w_wp_cond_ent':[], 'fp_W': [], 'fp_Wp':[], 'fp_W_conc_mean': [], 'fp_W_conc_sem': [],'fp_Wp_conc_mean': [], 'fp_Wp_conc_sem': [], 'fp_W_mean':[], 'fp_W_sem':[], 'fp_p_t1W_mean':[], 'fp_Wp_t1_sem':[],'fp_Wp_t2_mean':[], 'fp_Wp_t2_sem':[], 'fp_Wp_t3_mean':[], 'fp_Wp_t3_sem':[], 'mean_purity':[], 'sem_purity':[], 'mean_concurrence':[], 'sem_concurrence':[], 'mean_W':[], 'sem_W':[], 'mean_Wp_t1':[], 'sem_Wp_t1':[], 'mean_Wp_t2':[], 'sem_Wp_t2':[], 'mean_Wp_t3':[], 'sem_Wp_t3':[]})

        fig, ax = plt.subplots(2, 6, figsize=(30, 10))
        for i, data in enumerate(data_ls):
            df = pd.read_csv(join(DATA_PATH, data))

            purity_ls = df['purity'].to_numpy()
            concurrence_ls = df['concurrence'].to_numpy()
            W_ls = df['W_min'].to_numpy()
            Wp_t1_ls = df['Wp_t1'].to_numpy()
            Wp_t2_ls = df['Wp_t2'].to_numpy()
            Wp_t3_ls = df['Wp_t3'].to_numpy()

            # get mean and sem of purity, concurrence, W, Wp_t1, Wp_t2, Wp_t3 #
            mean_purity = np.mean(purity_ls)
            sem_purity = sem(purity_ls)
            mean_concurrence = np.mean(concurrence_ls)
            sem_concurrence = sem(concurrence_ls)
            mean_W = np.mean(W_ls)
            sem_W = sem(W_ls)
            mean_Wp_t1 = np.mean(Wp_t1_ls)
            sem_Wp_t1 = sem(Wp_t1_ls)
            mean_Wp_t2 = np.mean(Wp_t2_ls)
            sem_Wp_t2 = sem(Wp_t2_ls)
            mean_Wp_t3 = np.mean(Wp_t3_ls)
            sem_Wp_t3 = sem(Wp_t3_ls)

            # get stats relating to entanglement #
            ent_df = df.loc[df['concurrence']>0]
            ent = len(ent_df) / len(df)
            # false positive
            fp_W_df = df.loc[(df['concurrence']==0) & (df['W_min']<0)]
            fp_Wp_df = df.loc[(df['concurrence']==0) & ((df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t3']<0)) ]

            fp_W_conc_mean = np.mean(fp_W_df['concurrence'].to_numpy())
            fp_W_conc_sem = sem(fp_W_df['concurrence'].to_numpy())
            fp_W_mean = np.mean(fp_W_df['W_min'].to_numpy())
            fp_W_sem = sem(fp_W_df['W_min'].to_numpy())
            fp_Wp_conc_mean = np.mean(fp_Wp_df['concurrence'].to_numpy())
            fp_Wp_conc_sem = sem(fp_Wp_df['concurrence'].to_numpy())
            fp_Wp_t1_mean = np.mean(fp_Wp_df['Wp_t1'].to_numpy())
            fp_Wp_t1_sem = sem(fp_Wp_df['Wp_t1'].to_numpy())
            fp_Wp_t2_mean = np.mean(fp_Wp_df['Wp_t2'].to_numpy())
            fp_Wp_t2_sem = sem(fp_Wp_df['Wp_t2'].to_numpy())
            fp_Wp_t3_mean = np.mean(fp_Wp_df['Wp_t3'].to_numpy())
            fp_Wp_t3_sem = sem(fp_Wp_df['Wp_t3'].to_numpy())


            fp_W = len(fp_W_df) / len(df)
            fp_Wp = len(fp_Wp_df) / len(df)

            # states undetected by W #
            undet_W_df = ent_df.loc[(ent_df['W_min']>=0)] 
            undet_Wp_df = ent_df.loc[(ent_df['Wp_t1']>=0) & (ent_df['Wp_t2']>=0) & (ent_df['Wp_t3']>=0)]
            undet_all_df = ent_df.loc[(ent_df['W_min']>=0) & (ent_df['Wp_t1']>=0) & (ent_df['Wp_t2']>=0) & (ent_df['Wp_t3']>=0)]

            def get_frac_undet_W(threshold=0):
                ''' Gets fraction of undetected states by W given concurrence threshold'''
                return len(undet_W_df.loc[undet_W_df['concurrence']>threshold])/len(ent_df)
            def get_frac_undet_Wp(threshold=0):
                ''' Gets fraction of undetected states by Wp witnesses given concurrence threshold'''
                return len(undet_Wp_df.loc[undet_Wp_df['concurrence']>threshold])/len(ent_df)
            def get_frac_undet_all(threshold=0):
                ''' Gets fraction of undetected states by all witnesses given concurrence threshold'''
                return len(undet_all_df.loc[undet_all_df['concurrence']>threshold])/len(ent_df)

            undet_W_0 = get_frac_undet_W()
            undet_Wp_0 = get_frac_undet_Wp()
            undet_all_0 = get_frac_undet_all()

            # states that satisfy W>=0 #
            w_cond_all = len(df.loc[df['W_min']>=0]) / len(df)
            w_cond_ent = len(ent_df.loc[ent_df['W_min']>=0]) / len(ent_df)
            # states that satisfy at least one Wp < 0 #
            w_wp_cond_all = len(df.loc[(df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t3']<0)]) / len(df)
            w_wp_cond_ent = len(ent_df.loc[(ent_df['W_min']>=0)&((ent_df['Wp_t1']<0) | (ent_df['Wp_t2']<0) | (ent_df['Wp_t3']<0))]) / len(ent_df)

            # plot subplots #
            ax[0, 0].hist(purity_ls, bins=20, alpha=.5, label=names[i])
            ax[0,0].set_title('Purity Distribution')
            ax[0,0].set_xlabel('Purity')
            ax[0,0].set_ylabel('Counts')
            ax[0,0].legend()
            ax[0, 1].hist(concurrence_ls, bins=20, alpha=.5, label=names[i])
            ax[0,1].set_title('Concurrence Distribution')
            ax[0,1].set_xlabel('Concurrence')
            ax[0,1].set_ylabel('Counts')
            ax[0,1].legend()
            ax[0, 2].hist(W_ls, bins=20, alpha=.5, label=names[i])
            ax[0,2].set_title('$W$ Distribution')
            ax[0,2].set_xlabel('$W$')
            ax[0,2].set_ylabel('Counts')
            ax[0,2].legend()
            ax[0,3].hist(undet_W_df['concurrence'].to_numpy(), bins=20, alpha=.5, label=names[i])
            ax[0,3].set_title('Undetected by $W$ Distribution')
            ax[0,3].set_xlabel('Concurrence')
            ax[0,3].set_ylabel('Counts')
            ax[0,3].legend()
            ax[0,4].hist(undet_Wp_df['concurrence'].to_numpy(), bins=20, alpha=.5, label=names[i])
            ax[0,4].set_title('Undetected by $W\'$ Distribution')
            ax[0,4].set_xlabel('Concurrence')
            ax[0,4].set_ylabel('Counts')
            ax[0,4].legend()
            ax[0,5].hist(undet_all_df['concurrence'].to_numpy(), bins=20, alpha=.5, label=names[i])
            ax[0,5].set_title("Undetected by all $W, W'$ Distribution")
            ax[0,5].set_xlabel('Concurrence')
            ax[0,5].set_ylabel('Counts')
            ax[0,5].legend()
            ax[1, 0].hist(Wp_t1_ls, bins=20, alpha=.5, label=names[i])
            ax[1,0].set_title("$W'_{t_1}$ Distribution")
            ax[1,0].set_xlabel("$W'_{t_1}$")
            ax[1,0].set_ylabel('Counts')
            ax[1,0].legend()
            ax[1, 1].hist(Wp_t2_ls, bins=20, alpha=.5, label=names[i])
            ax[1,1].set_title("$W'_{t_2}$ Distribution")
            ax[1,1].set_xlabel("$W'_{t_2}$")
            ax[1,1].set_ylabel('Counts')
            ax[1,1].legend()
            ax[1, 2].hist(Wp_t3_ls, bins=20, alpha=.5, label=names[i])
            ax[1,2].set_title("$W'_{t_3}$ Distribution")
            ax[1,2].set_xlabel("$W'_{t_3}$")
            ax[1,2].set_ylabel('Counts')
            ax[1,2].legend()
            # get undetec frac list
            undet_W_frac_ls = [get_frac_undet_W(threshold) for threshold in np.arange(0, .25, .05)]
            ax[1,3].plot(np.arange(0, .25, .05), undet_W_frac_ls, label=names[i])
            ax[1,3].set_title("Fraction Undetected by $W$")
            ax[1,3].set_xlabel('Concurrence Threshold')
            ax[1,3].set_ylabel('Fraction Undetected')
            ax[1,3].legend()
            undet_Wp_frac_ls = [get_frac_undet_Wp(threshold) for threshold in np.arange(0, .25, .05)]
            ax[1,4].plot(np.arange(0, .25, .05), undet_Wp_frac_ls, label=names[i])
            ax[1,4].set_title("Fraction Undetected by $W'$")
            ax[1,4].set_xlabel('Concurrence Threshold')
            ax[1,4].set_ylabel('Fraction Undetected')
            ax[1,4].legend()
            undet_all_frac_ls = [get_frac_undet_all(threshold) for threshold in np.arange(0, .25, .05)]
            ax[1,5].plot(np.arange(0, .25, .05), undet_all_frac_ls, label=names[i])
            ax[1,5].set_title("Fraction Undetected by all $W, W'$")
            ax[1,5].set_xlabel('Concurrence Threshold')
            ax[1,5].set_ylabel('Fraction Undetected')
            ax[1,5].legend()

            # update info_df
            info_df = pd.concat([info_df, pd.DataFrame.from_records([{'name':names[i], 'ent':ent, 'undet_W_0':undet_W_0, 'undet_Wp_0': undet_all_0, 'undet_all_0':undet_all_0, 'w_cond_ent':w_cond_ent, 'w_wp_cond_ent':w_wp_cond_ent, 'fp_W': fp_W, 'fp_Wp':fp_Wp, 'fp_W_conc_mean': fp_W_conc_mean, 'fp_W_conc_sem': fp_W_conc_sem,'fp_Wp_conc_mean': fp_Wp_conc_mean, 'fp_Wp_conc_sem': fp_Wp_conc_sem, 'fp_W_mean':fp_W_mean, 'fp_W_sem':fp_W_sem, 'fp_p_t1W_mean':fp_Wp_t1_mean, 'fp_Wp_t1_sem':fp_Wp_t1_sem,'fp_Wp_t2_mean':fp_Wp_t2_mean, 'fp_Wp_t2_sem':fp_Wp_t2_sem, 'fp_Wp_t3_mean':fp_Wp_t3_mean, 'fp_Wp_t3_sem':fp_Wp_t3_sem, 'mean_purity':mean_purity, 'sem_purity':sem_purity, 'mean_concurrence':mean_concurrence, 'sem_concurrence':sem_concurrence, 'mean_W':mean_W, 'sem_W':sem_W, 'mean_Wp_t1':mean_Wp_t1, 'sem_Wp_t1':sem_Wp_t1, 'mean_Wp_t2':mean_Wp_t2, 'sem_Wp_t2':sem_Wp_t2, 'mean_Wp_t3':mean_Wp_t3, 'sem_Wp_t3':sem_Wp_t3 }])])


        plt.suptitle(titlename)
        plt.tight_layout()
        plt.savefig(join(DATA_PATH, savename + '.pdf'))
        plt.show()
        info_df.to_csv(join(DATA_PATH, savename + '_info.csv'), index=False)
    else:
        fig, ax = plt.subplots(2, 3, figsize=(30, 10))
        for i, data in enumerate(data_ls):
            df = pd.read_csv(join(DATA_PATH, data))

            purity_ls = df['purity'].to_numpy()
            concurrence_ls = df['concurrence'].to_numpy()
            W_ls = df['W_min'].to_numpy()
            Wp_t1_ls = df['Wp_t1'].to_numpy()
            Wp_t2_ls = df['Wp_t2'].to_numpy()
            Wp_t3_ls = df['Wp_t3'].to_numpy()

            # get mean and sem of purity, concurrence, W, Wp_t1, Wp_t2, Wp_t3 #
            mean_purity = np.mean(purity_ls)
            sem_purity = sem(purity_ls)
            mean_concurrence = np.mean(concurrence_ls)
            sem_concurrence = sem(concurrence_ls)
            mean_W = np.mean(W_ls)
            sem_W = sem(W_ls)
            mean_Wp_t1 = np.mean(Wp_t1_ls)
            sem_Wp_t1 = sem(Wp_t1_ls)
            mean_Wp_t2 = np.mean(Wp_t2_ls)
            sem_Wp_t2 = sem(Wp_t2_ls)
            mean_Wp_t3 = np.mean(Wp_t3_ls)
            sem_Wp_t3 = sem(Wp_t3_ls)

            # get stats relating to entanglement #
            ent_df = df.loc[df['concurrence']>0]
            ent = len(ent_df) / len(df)
            print(ent)






# data_ls = ['hurwitz_True_10000_stokesW_method_1.csv','hurwitz_True_10000_operatorW_method_1.csv']
# names = ['Stokes', 'Operator']
# savename = 'stokes_op_comp'
# titlename = 'Comparison of Stokes and Operator Witness Calculations for 10,000 States'

# data_ls = ['hurwitz_True_100000_b0_method_1_old2.csv', 'hurwitz_True_100000_correct_stokes_method_1.csv', 'hurwitz_True_100000_operatorW_method_1.csv']
# names=['Old Stokes', 'New Stokes', 'Operator']
# savename='stokes_old_op_comp'
# titlename = 'Comparison of Stokes and Operator Witness Calculations for 100,000 States'

data_ls =['hurwitz_True_1000_opt_method_1.csv', 'matlab_states.csv']
names = ['Hurwitz', 'Matlab']
savename = 'mat_comp_1'
titlename = 'Comparison of Witness Calculations for 1,000 States'

# comp_info(data_ls, names, savename, titlename)

def get_random_dist(mfile):
    '''Plots np.random.rand dist against matlab'''
    mrand = sio.loadmat(join(DATA_PATH, mfile+'.mat'))[mfile]
    print(mrand)
    mrand = mrand.flatten()
    np_rand = np.random.rand(len(mrand))*2*np.pi
    plt.hist(mrand, bins=100, alpha=0.5, label='Matlab')
    plt.hist(np_rand, bins=100, alpha=0.5, label='Numpy')
    plt.xlabel('Random Number')
    plt.ylabel('Frequency')
    plt.title('Comparison of Random Number Distributions')
    plt.legend()
    plt.savefig(mfile+'_np_comp.pdf')
    plt.show()

get_random_dist('unifrand_ls')



## plot of concurrence of werner ##
def werner_plot():
    p_ls = np.linspace(0, 1, 10)
    conc_ls = [get_concurrence(ws) for ws in [get_werner_state(p) for p in p_ls]]
    plt.plot(p_ls, conc_ls)
    plt.show()

## plots based on summer 2022 ##
def get_W2_ads():
    ''' Plots W2 expec value for amplitude damped states'''
    gamma_ls = np.linspace(0.3, 1, 100)
    W_ls = np.array([compute_witnesses(get_ads(gamma), ads_test=True) for gamma in gamma_ls])

    fig, ax = plt.subplots(2,1, figsize=(5,10), sharex=True)
    ax[0].plot(gamma_ls, W_ls[:, 0])
    ax[0].set_ylabel('$\langle W_2 \\rangle$')
    ax[1].plot(gamma_ls, np.sin(W_ls[:, 1]))
    ax[1].set_xlabel('$\gamma$')
    ax[1].set_ylabel('$\sin(\\theta)$')
    plt.suptitle('W2 Expectation Value for Amplitude Damped States')
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, 'w_test', 'w2_ads.pdf'))
    plt.show()

def comp_ex1():
    ''' Plot min W, W' values for the state cos(phi) |HD> - sin(phi) |VD> '''
    phi_ls = np.linspace(0, np.pi/2, 100)
    min_W_ls = np.array([compute_witnesses(get_ex1(phi), do_stokes=False) for phi in phi_ls])
    print(min_W_ls)
    W = min_W_ls[:, 0]
    Wp = np.amin(min_W_ls[:, 1:], axis=1)
    plt.plot(phi_ls, W, label='$W$')
    plt.plot(phi_ls, Wp, label='$W\'$')
    plt.xlabel('$\phi$')
    plt.ylabel('Min Witness Value')
    plt.title('Min witness value for ex1')
    plt.legend()
    plt.savefig(join(DATA_PATH, 'w_test', 'ex1_x0_rand10.pdf'))
    plt.show()

## process matlab data ##
def process_matlab_data_raw():
    from tqdm import tqdm
    states = sio.loadmat(join(DATA_PATH, 'states.mat'))['states']
    np_states = [np.array(state) for state in states[0]]
    state_df = pd.DataFrame({'W_min':[], 'Wp_t1':[], 'Wp_t2':[], 'Wp_t3':[], 'concurrence':[], 'purity':[]})
    for state in tqdm(np_states):
        conc = get_concurrence(state)
        purity = get_purity(state)
        W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(state)
        state_df = pd.concat([state_df, pd.DataFrame.from_records([{'W_min':W_min, 'Wp_t1':Wp_t1, 'Wp_t2':Wp_t2, 'Wp_t3':Wp_t3, 'concurrence':conc, 'purity':purity}])])
    state_df.to_csv(join(DATA_PATH, 'matlab_states.csv'), index=False)

def read_matlab_processed():
    import scipy.io as sio
    w_vals = sio.loadmat(join(DATA_PATH, 'w_vals.mat'))
    print(w_vals)

# process_matlab_data()
# read_matlab_processed()
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

def comp_info(data_ls, names, savename, titlename, do_entangled):
    ''' Function to plot properties of states and witness values, including the states UD by the witnesses. 
    
    params:
        data_ls: list of csvs to read in
        names: list of names of data
        savename: name to save figure as
        titlename: title of figure
        do_entangled: bool, whether to plot entangled states
    '''

    # read in data and calculate quantities
    # have df to store info about states
    info_df = pd.DataFrame({'name':[], 'ent':[], 'undet_W_0':[], 'undet_Wp_0':[], 'w_cond_ent':[], 'w_wp_cond_ent':[], 'fp_W': [], 'fp_Wp':[], 'fp_W_conc_mean': [], 'fp_W_conc_sem': [],'fp_Wp_conc_mean': [], 'fp_Wp_conc_sem': [], 'fp_W_mean':[], 'fp_W_sem':[], 'fp_p_t1W_mean':[], 'fp_Wp_t1_sem':[],'fp_Wp_t2_mean':[], 'fp_Wp_t2_sem':[], 'fp_Wp_t3_mean':[], 'fp_Wp_t3_sem':[], 'mean_purity':[], 'sem_purity':[], 'mean_concurrence':[], 'sem_concurrence':[], 'mean_W':[], 'sem_W':[], 'mean_Wp_t1':[], 'sem_Wp_t1':[], 'mean_Wp_t2':[], 'sem_Wp_t2':[], 'mean_Wp_t3':[], 'sem_Wp_t3':[]})

    # fig1, ax = plt.subplots(4, 3) # plots for all states
    if not(do_entangled):
        fig, ax = plt.subplots(4, 3, figsize=(10,15)) # plots for all
    else:
        fig, ax = plt.subplots(6, 3, figsize=(10,15))
    for i, data in enumerate(data_ls):
        df = pd.read_csv(join(DATA_PATH, data))

        
        if not('concurrence_c' in df.columns):
            purity_ls = df['purity'].to_numpy()
            purity_ent_ls = df.loc[df['concurrence']>0]['purity'].to_numpy() # purity of entangled states
            concurrence_ls = df['concurrence'].to_numpy()
            concurrence_ent_ls = df.loc[df['concurrence']>0]['concurrence'].to_numpy() # concurrence of entangled states
            # min_eig_ls = df['min_eig'].to_numpy()
            # min_eig_ent_ls = df.loc[df['concurrence']>0]['min_eig'].to_numpy() # min eig of entangled states
            alpha_ls = df['alpha'].to_numpy()
            alpha_ent_ls = df.loc[df['concurrence']>0]['alpha'].to_numpy() # alpha of entangled states
            psi_ls = df['psi'].to_numpy()
            psi_ent_ls = df.loc[df['concurrence']>0]['psi'].to_numpy() # psi of entangled states
            chi_ls = df['chi'].to_numpy()
            chi_ent_ls = df.loc[df['concurrence']>0]['chi'].to_numpy() # chi of entangled states
            phi_ls = df['phi'].to_numpy()
            phi_ent_ls = df.loc[df['concurrence']>0]['phi'].to_numpy() # phi of entangled states
            W_ls = df['W_min'].to_numpy()
            W_ent_ls = df.loc[df['concurrence']>0]['W_min'].to_numpy() # W of entangled states
            Wp_t1_ls = df['Wp_t1'].to_numpy()
            Wp_t1_ent_ls = df.loc[df['concurrence']>0]['Wp_t1'].to_numpy() # Wp_t1 of entangled states
            Wp_t2_ls = df['Wp_t2'].to_numpy()
            Wp_t2_ent_ls = df.loc[df['concurrence']>0]['Wp_t2'].to_numpy() # Wp_t2 of entangled states
            Wp_t3_ls = df['Wp_t3'].to_numpy()
            Wp_t3_ent_ls = df.loc[df['concurrence']>0]['Wp_t3'].to_numpy() # Wp_t3 of entangled states


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

            # states UD by W #
            undet_W_df = ent_df.loc[(ent_df['W_min']>=0)] 
            undet_Wp_df = ent_df.loc[(ent_df['Wp_t1']>=0) & (ent_df['Wp_t2']>=0) & (ent_df['Wp_t3']>=0)]

            def get_frac_undet_W(threshold=0):
                ''' Gets fraction of UD states by W given concurrence threshold'''
                return len(undet_W_df.loc[undet_W_df['concurrence']>threshold])/len(ent_df)
            def get_frac_undet_Wp(threshold=0):
                ''' Gets fraction of UD states by Wp witnesses given concurrence threshold'''
                return len(undet_Wp_df.loc[undet_Wp_df['concurrence']>threshold])/len(ent_df)
        else:
            purity_ls = df['purity'].to_numpy()
            purity_ent_ls = df.loc[df['concurrence_c']>0]['purity'].to_numpy() # purity of entangled states
    
            concurrence_ls = df['concurrence_c'].to_numpy()
            concurrence_ent_ls = df.loc[df['concurrence_c']>0]['concurrence_c'].to_numpy() # concurrence of entangled states
            # min_eig_ls = df['min_eig'].to_numpy()
            # min_eig_ent_ls = df.loc[df['concurrence']>0]['min_eig'].to_numpy() # min eig of entangled states
            alpha_ls = df['alpha'].to_numpy()
            alpha_ent_ls = df.loc[df['concurrence_c']>0]['alpha'].to_numpy() # alpha of entangled states
            psi_ls = df['psi'].to_numpy()
            psi_ent_ls = df.loc[df['concurrence_c']>0]['psi'].to_numpy() # psi of entangled states
            chi_ls = df['chi'].to_numpy()
            chi_ent_ls = df.loc[df['concurrence_c']>0]['chi'].to_numpy() # chi of entangled states
            phi_ls = df['phi'].to_numpy()
            phi_ent_ls = df.loc[df['concurrence_c']>0]['phi'].to_numpy() # phi of entangled states
            W_ls = df['W_min'].to_numpy()
            W_ent_ls = df.loc[df['concurrence_c']>0]['W_min'].to_numpy() # W of entangled states
            Wp_t1_ls = df['Wp_t1'].to_numpy()
            Wp_t1_ent_ls = df.loc[df['concurrence_c']>0]['Wp_t1'].to_numpy() # Wp_t1 of entangled states
            Wp_t2_ls = df['Wp_t2'].to_numpy()
            Wp_t2_ent_ls = df.loc[df['concurrence_c']>0]['Wp_t2'].to_numpy() # Wp_t2 of entangled states
            Wp_t3_ls = df['Wp_t3'].to_numpy()
            Wp_t3_ent_ls = df.loc[df['concurrence_c']>0]['Wp_t3'].to_numpy() # Wp_t3 of entangled states


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
            ent_df = df.loc[df['concurrence_c']>0]
            ent = len(ent_df) / len(df)

            # states UD by W #
            undet_W_df = ent_df.loc[(ent_df['W_min']>=0)] 
            undet_Wp_df = ent_df.loc[(ent_df['Wp_t1']>=0) & (ent_df['Wp_t2']>=0) & (ent_df['Wp_t3']>=0)]

            def get_frac_undet_W(threshold=0):
                ''' Gets fraction of UD states by W given concurrence threshold'''
                return len(undet_W_df.loc[undet_W_df['concurrence_c']>threshold])/len(ent_df)
            def get_frac_undet_Wp(threshold=0):
                ''' Gets fraction of UD states by Wp witnesses given concurrence threshold'''
                return len(undet_Wp_df.loc[undet_Wp_df['concurrence_c']>threshold])/len(ent_df)
        

        undet_W_0 = get_frac_undet_W()
        undet_Wp_0 = get_frac_undet_Wp()

        conc_threshold_ls = np.linspace(0, .25, 5)
        frac_ud_W_ls = [get_frac_undet_W(threshold) for threshold in conc_threshold_ls]
        frac_ud_Wp_ls = [get_frac_undet_Wp(threshold) for threshold in conc_threshold_ls]

        # states that satisfy W>=0 #
        w_cond_all = len(df.loc[df['W_min']>=0]) / len(df)
        w_cond_ent = len(ent_df.loc[ent_df['W_min']>=0]) / len(ent_df)
        # states that satisfy at least one Wp < 0 #
        w_wp_cond_all = len(df.loc[(df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t3']<0)]) / len(df)
        w_wp_cond_ent = len(ent_df.loc[(ent_df['W_min']>=0)&((ent_df['Wp_t1']<0) | (ent_df['Wp_t2']<0) | (ent_df['Wp_t3']<0))]) / len(ent_df)

        if not(do_entangled):

            # plot subplots #
            ax[0,0].hist(purity_ls, bins=20, alpha=.5, label=names[i])
            ax[0,0].set_title('Purity Distribution')
            ax[0,0].set_xlabel('Purity')
            ax[0,0].set_ylabel('Counts')
            ax[0,0].legend()
            ax[0,1].hist(concurrence_ls, bins=20, alpha=.5, label=names[i])
            ax[0,1].set_title('Concurrence Distribution')
            ax[0,1].set_xlabel('Concurrence')
            ax[0,1].set_ylabel('Counts')
            ax[0,1].legend()
            # ax[0,2].hist(min_eig_ls, bins=20, alpha=.5, label=names[i])
            # ax[0,2].set_title('Minimum $\lambda_{\\text{PT}}$ Distribution')
            # ax[0,2].set_xlabel('$\lambda_{\\text{PT}}$')
            # ax[0,2].set_ylabel('Counts')
            # ax[0,2].legend()
            ax[1,0].hist(alpha_ls, bins=20, alpha=.5, label=names[i])
            ax[1,0].set_title('$\\alpha$ Distribution')
            ax[1,0].set_xlabel('$\\alpha$')
            ax[1,0].set_ylabel('Counts')
            ax[1,0].legend()
            ax[1,1].hist(psi_ls, bins=20, alpha=.5, label=names[i])
            ax[1,1].set_title('$\\psi$ Distribution')
            ax[1,1].set_xlabel('$\\psi$')
            ax[1,1].set_ylabel('Counts')
            ax[1,1].legend()
            ax[1,2].hist(chi_ls, bins=20, alpha=.5, label=names[i])
            ax[1,2].set_title('$\\chi$ Distribution')
            ax[1,2].set_xlabel('$\\chi$')
            ax[1,2].set_ylabel('Counts')
            ax[1,2].legend()
            ax[2,0].hist(phi_ls, bins=20, alpha=.5, label=names[i])
            ax[2,0].set_title('$\\phi$ Distribution')
            ax[2,0].set_xlabel('$\\phi$')
            ax[2,0].set_ylabel('Counts')
            ax[2,0].legend()
            ax[2,1].hist(W_ls, bins=20, alpha=.5, label=names[i])
            ax[2,1].set_title('$W$ Distribution')
            ax[2,1].set_xlabel('$W$')
            ax[2,1].set_ylabel('Counts')
            ax[2,1].legend()
            ax[3,0].hist(Wp_t1_ls, bins=20, alpha=.5, label=names[i])
            ax[3,0].set_title('$W\'_{t1}$ Distribution')
            ax[3,0].set_xlabel('$W\'_{t1}$')
            ax[3,0].set_ylabel('Counts')
            ax[3,0].legend()
            ax[3,1].hist(Wp_t2_ls, bins=20, alpha=.5, label=names[i])
            ax[3,1].set_title('$W\'_{t2}$ Distribution')
            ax[3,1].set_xlabel('$W\'_{t2}$')
            ax[3,1].set_ylabel('Counts')
            ax[3,1].legend()
            ax[3,2].hist(Wp_t3_ls, bins=20, alpha=.5, label=names[i])
            ax[3,2].set_title('$W\'_{t3}$ Distribution')
            ax[3,2].set_xlabel('$W\'_{t3}$')
            ax[3,2].set_ylabel('Counts')
            ax[3,2].legend()
        ## entangled ##
        else:
            ax[0,0].hist(purity_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[0,0].set_title('Purity Entangled Distribution')
            ax[0,0].set_xlabel('Purity')
            ax[0,0].set_ylabel('Counts')
            ax[0,0].legend()
            ax[0,1].hist(concurrence_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[0,1].set_title('Concurrence Entangled Distribution')
            ax[0,1].set_xlabel('Concurrence')
            ax[0,1].set_ylabel('Counts')
            ax[0,1].legend()
            # ax[4,2].hist(min_eig_ent_ls, bins=20, alpha=.5, label=names[i])
            # ax[4,2].set_title('Minimum $\lambda_{\\text{PT}}$ Entangled Distribution')
            # ax[4,2].set_xlabel('$\lambda_{\\text{PT}}$')
            # ax[4,2].set_ylabel('Counts')
            # ax[4,2].legend()
            ax[1,0].hist(alpha_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[1,0].set_title('$\\alpha$ Entangled Distribution')
            ax[1,0].set_xlabel('$\\alpha$')
            ax[1,0].set_ylabel('Counts')
            ax[1,0].legend()
            ax[1,1].hist(psi_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[1,1].set_title('$\\psi$ Entangled Distribution')
            ax[1,1].set_xlabel('$\\psi$')
            ax[1,1].set_ylabel('Counts')
            ax[1,1].legend()
            ax[1,2].hist(chi_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[1,2].set_title('$\\chi$ Entangled Distribution')
            ax[1,2].set_xlabel('$\\chi$')
            ax[1,2].set_ylabel('Counts')
            ax[1,2].legend()
            ax[2,0].hist(phi_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[2,0].set_title('$\\phi$ Entangled Distribution')
            ax[2,0].set_xlabel('$\\phi$')
            ax[2,0].set_ylabel('Counts')
            ax[2,0].legend()
            ax[2,1].hist(W_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[2,1].set_title('$W$ Entangled Distribution')
            ax[2,1].set_xlabel('$W$')
            ax[2,1].set_ylabel('Counts')
            ax[2,1].legend()
            ax[3,0].hist(Wp_t1_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[3,0].set_title('$W\'_{t1}$ Entangled Distribution')
            ax[3,0].set_xlabel('$W\'_{t1}$')
            ax[3,0].set_ylabel('Counts')
            ax[3,0].legend()
            ax[3,1].hist(Wp_t2_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[3,1].set_title('$W\'_{t2}$ Entangled Distribution')
            ax[3,1].set_xlabel('$W\'_{t2}$')
            ax[3,1].set_ylabel('Counts')
            ax[3,1].legend()
            ax[3,2].hist(Wp_t3_ent_ls, bins=20, alpha=.5, label=names[i])
            ax[3,2].set_title('$W\'_{t3}$ Entangled Distribution')
            ax[3,2].set_xlabel('$W\'_{t3}$')
            ax[3,2].set_ylabel('Counts')
            ax[3,2].legend()
            ax[4,0].hist(undet_W_df['W_min'].to_numpy(), bins=20, alpha=.5, label=names[i])
            ax[4,0].set_title('UD $W$ Value Distribution')
            ax[4,0].set_xlabel('$W$')
            ax[4,0].set_ylabel('Counts')
            ax[4,0].legend()
            if 'concurrence_b' not in undet_W_df.columns:
                
                ax[4,1].hist(undet_W_df['concurrence'].to_numpy(), bins=20, alpha=.5, label=names[i])
                ax[4,1].set_title('UD $W$ Concurrence Value Distribution')
                ax[4,1].set_xlabel('Concurrence')
                ax[4,1].set_ylabel('Counts')
                ax[4,1].legend()
                # ax[4,2].hist(undet_W_df['min_eig'].to_numpy(), bins=20, alpha=.5, label=names[i])
                # ax[4,2].set_title('UD $W$ Minimum $\lambda_{\\text{PT}}$ Value Distribution')
                # ax[4,2].set_xlabel('$\lambda_{\\text{PT}}$')
                # ax[4,2].set_ylabel('Counts')
                # ax[4,2].legend()
                ax[4,2].hist(undet_Wp_df['concurrence'].to_numpy(), bins=20, alpha=.5, label=names[i])
                ax[4,2].set_title('UD $W\'$ Concurrence Value Distribution')
                ax[4,2].set_xlabel('Concurrence')
                ax[4,2].set_ylabel('Counts')
                ax[4,2].legend()
        
            else:
                ax[4,1].hist(undet_W_df['concurrence_c'].to_numpy(), bins=20, alpha=.5, label=names[i])
                ax[4,1].set_title('UD $W$ Concurrence Value Distribution')
                ax[4,1].set_xlabel('Concurrence')
                ax[4,1].set_ylabel('Counts')
                ax[4,1].legend()
                # ax[4,2].hist(undet_W_df['min_eig'].to_numpy(), bins=20, alpha=.5, label=names[i])
                # ax[4,2].set_title('UD $W$ Minimum $\lambda_{\\text{PT}}$ Value Distribution')
                # ax[4,2].set_xlabel('$\lambda_{\\text{PT}}$')
                # ax[4,2].set_ylabel('Counts')
                # ax[4,2].legend()
                ax[4,2].hist(undet_Wp_df['concurrence_c'].to_numpy(), bins=20, alpha=.5, label=names[i])
                ax[4,2].set_title('UD $W\'$ Concurrence Value Distribution')
                ax[4,2].set_xlabel('Concurrence')
                ax[4,2].set_ylabel('Counts')
                ax[4,2].legend()
            # ax[5,2].hist(undet_Wp_df['min_eig'].to_numpy(), bins=20, alpha=.5, label=names[i])
            # ax[5,2].set_title('UD $W\'$ Minimum $\lambda_{\\text{PT}}$ Value Distribution')
            # ax[5,2].set_xlabel('$\lambda_{\\text{PT}}$')
            # ax[5,2].set_ylabel('Counts')
            # ax[5,2].legend()
            ax[5,0].plot(conc_threshold_ls, frac_ud_W_ls, label=names[i])
            ax[5,0].set_title('Fraction of UD $W$')
            ax[5,0].set_xlabel('Concurrence Threshold')
            ax[5,0].set_ylabel('Fraction UD')
            ax[5,0].legend()
            ax[5,1].plot(conc_threshold_ls, frac_ud_Wp_ls, label=names[i])
            ax[5,1].set_title('Fraction of UD $W\'$')
            ax[5,1].set_xlabel('Concurrence Threshold')
            ax[5,1].set_ylabel('Fraction UD')
            ax[5,1].legend()

        # update info_df
        info_df = pd.concat([info_df, pd.DataFrame.from_records([{'name':names[i], 'ent':ent, 'undet_W_0':undet_W_0, 'undet_Wp_0': undet_Wp_0, 'w_cond_ent':w_cond_ent, 'w_wp_cond_ent':w_wp_cond_ent, 'mean_purity':mean_purity, 'sem_purity':sem_purity, 'mean_concurrence':mean_concurrence, 'sem_concurrence':sem_concurrence, 'mean_W':mean_W, 'sem_W':sem_W, 'mean_Wp_t1':mean_Wp_t1, 'sem_Wp_t1':sem_Wp_t1, 'mean_Wp_t2':mean_Wp_t2, 'sem_Wp_t2':sem_Wp_t2, 'mean_Wp_t3':mean_Wp_t3, 'sem_Wp_t3':sem_Wp_t3 }])])

    plt.suptitle(titlename)
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, savename + '.pdf'))
    # plt.show()
    info_df.to_csv(join(DATA_PATH, savename + '_info.csv'), index=False)

def comp_undetected(data_ls, names, savename, titlename):
    ''' Function to just compare the undetected fraction of states and distr of concurrence'''

    fig, ax = plt.subplots(1,3, figsize=(10,5))
    for i in range(len(data_ls)):
        df = pd.read_csv(join(DATA_PATH, data_ls[i]))
        # print(df.head())
        if 'concurrence' in df.columns: 
            ent_df = df.loc[df['concurrence']>0]
    
            print('num entangled', len(ent_df))
            undet_W_df = ent_df.loc[ent_df['W_min']>=0]
            print('percent W undetected', len(undet_W_df) / len(ent_df))
            undet_Wp_df = ent_df.loc[(ent_df['Wp_t1']>=0) & (ent_df['Wp_t2']>=0) & (ent_df['Wp_t3']>=0)]
            print('percent Wp undetected', len(undet_Wp_df) / len(ent_df))

            def get_frac_undet_W(threshold=0):
                ''' Gets fraction of UD states by W given concurrence threshold'''
                return len(undet_W_df.loc[undet_W_df['concurrence']>threshold])/len(ent_df)
            def get_frac_undet_Wp(threshold=0):
                ''' Gets fraction of UD states by Wp witnesses given concurrence threshold'''
                return len(undet_Wp_df.loc[undet_Wp_df['concurrence']>threshold])/len(ent_df)

            conc_threshold_ls = np.linspace(0,.25,5)
            frac_ud_W_ls = [get_frac_undet_W(threshold) for threshold in conc_threshold_ls]
            frac_ud_Wp_ls = [get_frac_undet_Wp(threshold) for threshold in conc_threshold_ls]

            ax[0].plot(conc_threshold_ls, frac_ud_W_ls, label=names[i])
            ax[0].set_title('Fraction of UD $W$')
            ax[0].set_ylabel('Fraction UD')
            ax[0].set_xlabel('Concurrence Threshold')
            ax[0].legend()
            ax[1].plot(conc_threshold_ls, frac_ud_Wp_ls, label=names[i])
            ax[1].set_title('Fraction of UD $W\'$')
            ax[1].set_ylabel('Fraction UD')
            ax[1].set_xlabel('Concurrence Threshold')
            ax[1].legend()
            ax[2].hist(ent_df['concurrence'].to_numpy(), bins=20, alpha=.5, label=names[i])
            ax[2].set_title('Entangled State Concurrence Distribution')
            ax[2].set_xlabel('Concurrence')
            ax[2].set_ylabel('Counts')
            ax[2].legend()
        else:
            ent_c_df = df.loc[df['concurrence_c']>0]
            ent_b_df = df.loc[df['concurrence_b']>0]
            print('frac ent correct', len(ent_c_df)/len(df))
            print('frac ent b', len(ent_b_df)/len(df))

            undet_W_c_df = ent_c_df.loc[ent_c_df['W_min']>=0]
            undet_W_b_df = ent_b_df.loc[ent_b_df['W_min']>=0]
            undet_Wp_c_df = ent_c_df.loc[(ent_c_df['Wp_t1']>=0) & (ent_c_df['Wp_t2']>=0) & (ent_c_df['Wp_t3']>=0)]
            undet_Wp_b_df = ent_b_df.loc[(ent_b_df['Wp_t1']>=0) & (ent_b_df['Wp_t2']>=0) & (ent_b_df['Wp_t3']>=0)]

            def get_frac_undet_W_c(threshold=0):
                ''' Gets fraction of UD states by W given concurrence threshold'''
                return len(undet_W_c_df.loc[undet_W_c_df['concurrence_c']>threshold])/len(ent_c_df)
            def get_frac_undet_W_b(threshold=0):
                ''' Gets fraction of UD states by Wp witnesses given concurrence threshold'''
                return len(undet_W_b_df.loc[undet_W_b_df['concurrence_b']>threshold])/len(ent_b_df)
            def get_frac_undet_Wp_c(threshold=0):
                ''' Gets fraction of UD states by Wp witnesses given concurrence threshold'''
                return len(undet_Wp_c_df.loc[undet_Wp_c_df['concurrence_c']>threshold])/len(ent_c_df)
            def get_frac_undet_Wp_b(threshold=0):
                ''' Gets fraction of UD states by Wp witnesses given concurrence threshold'''
                return len(undet_Wp_b_df.loc[undet_Wp_b_df['concurrence_b']>threshold])/len(ent_b_df)

            conc_threshold_ls = np.linspace(0,.25,5)
            frac_ud_W_c_ls = [get_frac_undet_W_c(threshold) for threshold in conc_threshold_ls]
            frac_ud_W_b_ls = [get_frac_undet_W_b(threshold) for threshold in conc_threshold_ls]
            frac_ud_Wp_c_ls = [get_frac_undet_Wp_c(threshold) for threshold in conc_threshold_ls]
            frac_ud_Wp_b_ls = [get_frac_undet_Wp_b(threshold) for threshold in conc_threshold_ls]

            ax[0].plot(conc_threshold_ls, frac_ud_W_c_ls, label=names[i]+'_c')
            ax[0].plot(conc_threshold_ls, frac_ud_W_b_ls, label=names[i]+'_b')
            ax[0].set_title('Fraction of UD $W$')
            ax[0].set_ylabel('Fraction UD')
            ax[0].set_xlabel('Concurrence Threshold')
            ax[0].legend()
            ax[1].plot(conc_threshold_ls, frac_ud_Wp_c_ls,  label=names[i]+'_c')
            ax[1].plot(conc_threshold_ls,frac_ud_Wp_b_ls, label=names[i]+'_b')
            ax[1].set_title('Fraction of UD $W\'$')
            ax[1].set_ylabel('Fraction UD')
            ax[1].set_xlabel('Concurrence Threshold')
            ax[1].legend()
            ax[2].hist(ent_c_df['concurrence_c'].to_numpy(), bins=20, alpha=.5, label=names[i]+'_c')
            ax[2].hist(ent_b_df['concurrence_b'].to_numpy(), bins=20, alpha=.5, label=names[i]+'_b')
            ax[2].set_title('Entangled State Concurrence Distribution')
            ax[2].set_xlabel('Concurrence')
            ax[2].set_ylabel('Counts')
            ax[2].legend()


    plt.suptitle(titlename)
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, savename + '.pdf'))


data_ls = ['10000_conc_comp_m.csv', 'hurwitz_True_10000_grad_method_1.csv', 'roik_True_10000_grad.csv']
names = ['Matlab', 'Me', 'Roik']
savename = 'mat_me_roik_comp_10000_grad'
titlename = 'Comparison of Concurrence Calculations for 10,000 States'
for do_entangled in [False, True]:
    comp_info(data_ls, names, savename+'_'+str(do_entangled), titlename, do_entangled)


# comp_undetected(data_ls, names, savename, titlename)

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
# helper file to get stats from jones_decomp files
import numpy as np
import pandas as pd
from scipy.stats import sem
from os.path import join

def get_stats(inputs, filename, savename):
    ''' Computes mean and sem of fidelity for particular Jones decomp. params:
    inputs: tuple of (state_key, setup, adapt)
        state_key: string of the state to analyze
        setup: 'C' or 'I'
        adapt: True or False
    '''
    fidelity_ls =[]
    fidelity_sem_ls = []
    n_ls = []
    n_sem_ls = []
    purity_ls = []
    purity_sem_ls = []
    for input in inputs:
        state_key, setup, adapt = input[0], input[1], input[2]
        # load the data
        df = pd.read_csv(filename)

        # print(len(df['state'].str.split('_').str[1][0][0][0]))
        # get the relevant data
        if state_key in ['PhiP', 'PhiM', 'PsiP', 'PsiM']:
            df = df.loc[(df['state']==state_key) & (df['setup']==setup) & (df['adapt']==adapt)]
        elif state_key in ['E0', 'E1', 'RS', 'roik']:
            df = df.loc[(df['state'].str.split('_').str[0]==state_key) & (df['setup']==setup) & (df['adapt']==adapt)]
        elif state_key=='jones_C':
            df=df.iloc[:201, :]
            print(df)
            df = df.loc[(df['state'].str.split('_').str[0]=='jones')  & (df['setup']==setup) & (df['adapt']==adapt)]
        elif state_key=='jones_I':
            df=df.iloc[201:401, :]
            print(df)
            df = df.loc[(df['state'].str.split('_').str[0]=='jones')  & (df['setup']==setup) & (df['adapt']==adapt)]
            print(df)

        # get fidelity
        fidelity = np.mean(df['fidelity'].to_numpy())
        fidelity_sem = sem(df['fidelity'].to_numpy())

        # get number of iterations
        n = np.mean(df['n'].to_numpy())
        n_sem = sem(df['n'].to_numpy())

        if state_key=='roik':
            print(df['state'].str.split('_').str[1])
            purity = np.mean(df['state'].str.split('_').str[1].to_numpy(dtype=float))
            print(purity)
            purity_sem = sem(df['state'].str.split('_').str[1].to_numpy(dtype=float))
            purity_ls.append(purity)
            purity_sem_ls.append(purity_sem)
        else:
            purity_ls.append(1)
            purity_sem_ls.append(0)

        fidelity_ls.append(fidelity)
        fidelity_sem_ls.append(fidelity_sem)

        n_ls.append(n)
        n_sem_ls.append(n_sem)

    out_df = pd.DataFrame()
    out_df['inputs'] = inputs
    out_df['n'] = n_ls
    out_df['n sem'] = n_sem_ls
    out_df['fidelity'] = fidelity_ls
    out_df['fidelity sem'] = fidelity_sem_ls
    if state_key=='roik':
        out_df['purity'] = purity_ls
        out_df['purity sem'] = purity_sem_ls
    out_df.to_csv(savename)
    return out_df


if __name__=='__main__':
    # filename = input('Filename to analyze?')
    # state_key = input('State to analyze?')
    # setup = input('Setup to analyze?')
    # adapt = input('Adapt to analyze?')
    def do_bell(file):
        bell_inputs = []
        for bell in ['PhiP', 'PhiM', 'PsiP', 'PsiM']:
            for setup in ['C','I']:
                    for adapt in [True,False]:
                        bell_inputs.append([bell, setup, adapt])
        bell_df = get_stats(bell_inputs, filename=join('decomp', file), savename=join('decomp', 'bell-stats_correctphi.csv'))
        print(bell_df)

    def do_er(file):
        other_inputs = []
        for state in ['E0', 'E1', 'RS']:
            for setup in ['C','I']:
                    # for adapt in [True,False]:
                    other_inputs.append([state, setup, False])
        other_df = get_stats(other_inputs, filename=join('decomp', file), savename=join('decomp', 'er-stats.csv'))
        print(other_df)

    def do_jr(file):
        other_inputs = []
        for state in ['jones_C', 'jones_I', 'roik']:
            for setup in ['C','I']:
                    # for adapt in [True,False]:
                    other_inputs.append([state, setup, False])
        other_df = get_stats(other_inputs, filename=join('decomp', file), savename=join('decomp', 'jr-stats.csv'))
        print(other_df)
    
    # do_bell('decomp_all_True_False_False_False_bell_phi_redo.csv')
    do_er('decomp_all_False_True_True_True_e_s_phi_redo.csv')
    # do_jr('decomp_all_False_False_False_False_j_r_1.csv')
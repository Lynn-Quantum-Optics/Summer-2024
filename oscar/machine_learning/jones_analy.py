# helper file to get stats from jones_decomp files
import numpy as np
import pandas as pd
from scipy.stats import sem
from os.path import join

def get_stats_bell(inputs, filename, savename):
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
    for input in inputs:
        state_key, setup, adapt = input[0], input[1], input[2]
        # load the data
        df = pd.read_csv(filename)
        # get the relevant data
        
        df = df.loc[(df['state']==state_key) & (df['setup']==setup) & (df['adapt']==adapt)]
        # get fidelity
        fidelity = np.mean(df['fidelity'].to_numpy())
        fidelity_sem = sem(df['fidelity'].to_numpy())

        n = np.mean(df['n'].to_numpy())
        n_sem = sem(df['n'].to_numpy())

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
    out_df.to_csv(savename)
    return out_df

if __name__=='__main__':
    # filename = input('Filename to analyze?')
    # state_key = input('State to analyze?')
    # setup = input('Setup to analyze?')
    # adapt = input('Adapt to analyze?')
    bell_inputs = []
    for bell in ['PhiP', 'PhiM', 'PsiP', 'PsiM']:
        for setup in ['C','I']:
                for adapt in [True,False]:
                    bell_inputs.append([bell, setup, adapt])
    bell_df = get_stats_bell(bell_inputs, filename=join('decomp', 'decomp_all_True_False_False_False_bell-c.csv'), savename=join('decomp', 'bell-stats.csv'))
# helper file to get stats from jones_decomp files
import numpy as np
import pandas as pd
from scipy.stats import sem
from os.path import join

def get_stats(inputs, filename, savename, split_index=None, end_index=None):
    ''' Computes mean and sem of fidelity for particular Jones decomp. params:
    inputs: tuple of (state_key, setup, adapt)
        state_key: string of the state to analyze
        setup: 'C' or 'I'
        adapt: True or False
    filename: the csv to analyze
    savename: the name of the file to save the data to
    split_index: the index to split the data at for jones_C vs jones_I; will be unnecessary once I change the way the files are named when they're created
    end_index: the index to end the data jones_I vs roik; will be unnecessary once I change the way the files are named when they're created
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
            assert split_index is not None, 'split_index must be specified for jones C'
            df=df.iloc[:split_index, :]
            df = df.loc[(df['state'].str.split('_').str[0]=='jones')  & (df['setup']==setup) & (df['adapt']==adapt)]
        elif state_key=='jones_I':
            assert split_index and end_index is not None, 'split_index and end index must be specified for jones I'
            df=df.iloc[split_index:end_index, :]
            df = df.loc[(df['state'].str.split('_').str[0]=='jones')  & (df['setup']==setup) & (df['adapt']==adapt)]

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
    def do_bell(file, outfile):
        bell_inputs = []
        for bell in ['PhiP', 'PhiM', 'PsiP', 'PsiM']:
            for setup in ['C','I']:
                    for adapt in [True,False]:
                        bell_inputs.append([bell, setup, adapt])
        bell_df = get_stats(bell_inputs, filename=join('decomp', file), savename=join('decomp', outfile+'.csv'))
        print(bell_df)

    def do_er(file, outfile):
        other_inputs = []
        for state in ['E0', 'E1', 'RS']:
            for setup in ['C','I']:
                    # for adapt in [True,False]:
                    other_inputs.append([state, setup, False])
        other_df = get_stats(other_inputs, filename=join('decomp', file), savename=join('decomp', outfile+'.csv'))
        print(other_df)

    def do_jr(file, outfile, split_index=None, end_index=None):
        other_inputs = []
        for state in ['jones_C', 'jones_I', 'roik']:
            for setup in ['C','I']:
                    # for adapt in [True,False]:
                    other_inputs.append([state, setup, False])
        other_df = get_stats(other_inputs, filename=join('decomp', file), savename=join('decomp', outfile+'.csv'), split_index=split_index, end_index=end_index)
        print(other_df)
    
    do_bell('decomp_all_True_False_False_False_bell_phi2.csv', outfile='bell-stats-phi2')
    # do_er('decomp_all_False_True_True_True_e_s_phi_redo.csv')
    # do_jr('decomp_all_False_False_False_False_j_r_phi_redo.csv', outfile='jr-stats-redo', split_index=100, end_index=200)
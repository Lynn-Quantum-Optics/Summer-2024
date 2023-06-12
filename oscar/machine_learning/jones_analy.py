# helper file to get stats from jones_decomp files
import numpy as np
import pandas as pd
from scipy.stats import sem

def get_stats(filename, state_key, setup, adapt):
    ''' Computes mean and sem of fidelity for particular Jones decomp. params:
    state_key: string of the state to analyze
    setup: 'C' or 'I'
    adapt: True or False
    '''

    # load the data
    df = pd.read_csv(filename)
    # get the relevant data
    df = df.loc[(df['state'].contains(state_key)) & (df['setup']==setup) & (df['adapt']==adapt)]
    # get fidelity
    fidelity = np.mean(df['fidelity'].to_numpy())
    # get sem
    sem = sem(df['fidelity'].to_numpy())

    return fidelity, sem

if __name__=='__main__':
    filename = input('Filename to analyze?')
    state_key = input('State to analyze?')
    setup = input('Setup to analyze?')
    adapt = input('Adapt to analyze?')
    fidelity, sem = get_stats(filename, state_key, setup, adapt)
    print('Fidelity: %f +/- %f'%(fidelity, sem))
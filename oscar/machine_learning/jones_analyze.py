# file to extract info from the random jones data
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load data generated from jones_datagen.py ##
DATA_PATH = 'jones_simplex_data'
df_jones = pd.read_csv(join(DATA_PATH, 'jones_50000_0.csv'))
df_simplex = pd.read_csv(join(DATA_PATH, 'simplex_50000_0.csv'))

## analyze the data ##
def get_min_eig_dist_comp(df_jones, df_simplex, savename=None):
    ''' Compare  the distribution of min eigenvalues for Jones and Simplex methods.'''
    min_eigs_jones = df_jones['min_eig']
    min_eigs_simplex = df_simplex['min_eig']
    
    fig, axes = plt.subplots(2,1, figsize=(7,8), sharex=True)
    axes[0].hist(min_eigs_jones, bins=100, color='purple', alpha=0.5)
    axes[0].set_title('Jones')
    axes[1].hist(min_eigs_simplex, bins=100, color='blue', alpha=0.50)
    axes[1].set_title('Simplex')
    axes[1].set_xlabel('Min eigenvalue')

    plt.suptitle('Distribution of min eigenvalues for 50000 random states')

    if savename != None:
        plt.savefig(join(DATA_PATH, savename+'.pdf'))
    plt.show()

# get_min_eig_dist_comp(df_jones, df_simplex, savename='min_eig_dist_comp_50000_0')
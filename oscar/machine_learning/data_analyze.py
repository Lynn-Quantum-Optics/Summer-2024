# file to extract info from the data we generated

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## analyze the data ##
def get_min_eig_dist_comp(df_ls, labels,title='sick title', savename=None):
    ''' Compare  the distribution of min eigenvalues for any number of dfs.'''
    # get min eigenvalues
    min_eig_ls = [df['min_eig'] for df in df_ls]
    
    fig, axes = plt.subplots(len(df_ls),1, figsize=(7,8), sharex=True)
    for i in range(len(df_ls)):
        axes[i].hist(min_eig_ls[i], bins=100, alpha=0.9)
        axes[i].set_title(labels[i])
        axes[i].set_ylabel('Counts')
    axes[-1].set_xlabel('Min eigenvalue')

    plt.suptitle('Distribution of min eigenvalues for %s'%title)

    if savename != None:
        plt.savefig(savename+'.pdf')
    plt.show()

    # return where min eig is negative
    return [df['min_eig'][df['min_eig']<0] for df in df_ls]

def get_purity_dist_comp(df_ls, labels, title='super cool title', savename=None):
    ''' Compare the distribution of purity for any number of dfs.'''
    # get purity
    purity_ls = [df['purity'] for df in df_ls]

    fig, axes = plt.subplots(len(df_ls),1, figsize=(7,8), sharex=True)
    for i in range(len(df_ls)):
        axes[i].hist(purity_ls[i], bins=100, alpha=0.9)
        axes[i].set_title(labels[i])
        axes[i].set_ylabel('Counts')
    axes[-1].set_xlabel('Purity')

    plt.suptitle('Distribution of purity for %s'%title)

    if savename != None:
        plt.savefig(savename+'.pdf')
    plt.show()

def get_witness_dist(df_ls, name, labels, plot=True, savename=None):
    ''' Compare the distribution of W values for any number of dfs.'''
    W_ls = [df['W_min'] for df in df_ls]
    Wp_t1_ls = [df['Wp_t1'] for df in df_ls]
    Wp_t2_ls = [df['Wp_t2'] for df in df_ls]
    Wp_t3_ls = [df['Wp_t3'] for df in df_ls]

    if plot:
        fig, axes = plt.subplots(1,4, figsize=(12,8), sharex=True, squeeze=False)
        for i in range(len(df_ls)):
            axes[0][0].hist(W_ls[i], bins=100, alpha=0.5, label=labels[i])
            axes[0][0].legend()
            axes[0][0].set_ylabel('Counts')
            axes[0][1].hist(Wp_t1_ls[i], bins=100, alpha=0.5, label=labels[i])
            axes[0][1].legend()
            axes[0][2].hist(Wp_t2_ls[i], bins=100, alpha=0.5, label=labels[i])
            axes[0][2].legend()
            axes[0][3].hist(Wp_t3_ls[i], bins=100, alpha=0.5, label=labels[i])
            axes[0][3].legend()
        
        plt.suptitle(f'Witness Distribution for {name}')
        axes[-1][0].set_xlabel('W')
        axes[-1][1].set_xlabel('Wp_t1')
        axes[-1][2].set_xlabel('Wp_t2')
        axes[-1][3].set_xlabel('Wp_t3')
        plt.savefig(savename+'.pdf')
        plt.show()

    # return: where W_min is negative, where W_min is non-negative but at least one of Wp_t1, Wp_t2, Wp_t3 is negative, where Wp_t1 is negative but W_min, Wp_t2 and Wpt_3 are nonnegative, where Wp_t2 is negative but W_min, Wp_t1 and Wpt_3 are nonnegative, where Wp_t3 is negative but W_min, Wp_t1 and Wpt_2 are nonnegative
    return [
        [df[df['W_min']<0] for df in df_ls],
        [df[(df['W_min']>=0) & ((df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t1']<0))] for df in df_ls], 
        [df[(df['Wp_t1']<0) & ((df['W_min']>=0) & (df['Wp_t2']>=0) & (df['Wp_t3']>=0))] for df in df_ls],
        [df[(df['Wp_t2']<0) & ((df['W_min']>=0) & (df['Wp_t1']>=0) & (df['Wp_t3']>=0))] for df in df_ls],
        [df[(df['Wp_t3']<0) & ((df['W_min']>=0) & (df['Wp_t1']>=0) & (df['Wp_t2']>=0))] for df in df_ls]
        ]


if __name__ == '__main__':
    DATA_PATH = 'random_gen/data'
    df_hurwitz0 = pd.read_csv(join(DATA_PATH, 'hurwitz_all_4400000_b0_method_2.csv'))

    # get_min_eig_dist_comp([df_jones, df_simplex], labels=['Jones', 'Simplex'],savename=join(DATA_PATH, 'min_eig_dist_comp_200000_0'))
    witness_ls  = get_witness_dist([df_hurwitz0], name='Hurtwitz', labels=['Method 0'], savename=join(DATA_PATH, 'witness_dist_comp_4400000_0'))

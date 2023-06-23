# file to compare the distributions of properties of random density matrices generated from my get_random_hurwitz() code and the code from the group in summer 2022/ fall 2022 / spring 2023

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join

# define list of csvs
DATA_PATH = 'random_gen/test'
data_ls = ['hurwitz_True_100000_test_old_method_1.csv', 'hurwitz_True_100000_b0_method_1_old2.csv', 'hurwitz_True_100000_test_upd_method_1.csv']
names = ['Old Method D1', 'Old Method D2', 'New Method']

savename = 'old_new_comp'
titlename = 'Comparison of Old and New Method for Generating Roik Unitaries for 100,000 States'

# read in data and caculate quantities
fig, ax = plt.subplots(2, 3, figsize=(15, 10), )
for i, data in enumerate(data_ls):
    df = pd.read_csv(join(DATA_PATH, data))

    purity_ls = df['purity'].to_numpy()
    concurrence_ls = df['concurrence'].to_numpy()
    W_ls = df['W_min'].to_numpy()
    Wp_t1_ls = df['Wp_t1'].to_numpy()
    Wp_t2_ls = df['Wp_t2'].to_numpy()
    Wp_t3_ls = df['Wp_t3'].to_numpy()

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

plt.suptitle(titlename)
plt.tight_layout()
plt.savefig(join(DATA_PATH, savename + '.pdf'))

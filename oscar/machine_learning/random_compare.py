# file to compare different random gneneration methods

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from random_gen import *

def compare_hurwitz(size=100):
    ''' compare my old def with roik and maziero implementation of hurwitz: in particular, how they deal with phi'''
    purity0_ls = []
    purity1_ls = []
    purity2_ls = []
    phi0_ls = []
    phi1_ls = []
    phi2_ls = []
    for i in trange(size):
        # get random density matrices and purity #
        rho0 = get_random_hurwitz(method=0)
        rho1 = get_random_hurwitz(method=1)
        rho2 = get_random_hurwitz(method=2)
        purity0_ls.append(get_purity(rho0))
        purity1_ls.append(get_purity(rho1))
        purity2_ls.append(get_purity(rho2))

        # get phi values #
        for k in range(5, -1, -1):
            phi0_ls.append(np.arcsin((np.random.rand())**1/(2*(k+1))))
            phi1_ls.append(np.arcsin((np.random.rand())**1/(2)))
            phi2_ls.append(np.random.rand()*np.pi/2)

    # plot purity subplots #
    # fig, ax = plt.subplots(3, 2, figsize=(15, 5))
    # ax[0, 0].hist(purity0_ls, bins=20)
    # ax[0, 0].set_title('Method 0')
    # ax[1, 0].hist(purity1_ls, bins=20)
    # ax[1, 0].set_title('Method 1')
    # ax[2, 0].hist(purity2_ls, bins=20)
    # ax[2, 0].set_title('Method 2')
    # ax[0, 1].hist(phi0_ls, bins=20)
    # ax[0, 1].set_title('Method 0')
    # ax[1, 1].hist(phi1_ls, bins=20)
    # ax[1, 1].set_title('Method 1')
    # ax[2, 1].hist(phi2_ls, bins=20)
    # ax[2, 1].set_title('Method 2')
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), )
    ax[0].hist(purity0_ls, bins=20, alpha=.5, color='red', label='Method 0')
    ax[0].hist(purity1_ls, bins=20, alpha=.5, color='blue', label='Method 1')
    ax[0].hist(purity2_ls, bins=20, alpha=.5, color='green', label='Method 2')
    ax[0].set_title('Purity Distribution')
    ax[0].set_xlabel('Purity')
    ax[0].legend()
    ax[1].hist(phi0_ls, bins=20, alpha=.5, color='red', label='Method 0')
    ax[1].hist(phi1_ls, bins=20, alpha=.5, color='blue', label='Method 1')
    ax[1].hist(phi2_ls, bins=20, alpha=.5, color='green', label='Method 2')
    ax[1].set_title('Phi Distruibution')
    ax[1].legend()
    ax[1].set_xlabel('Angle (Rad)')
    plt.title(f'Comparison of Hurwitz Method for {size} Random Density Matrices')
    plt.savefig(join('random_compare', f'hurwitz_compare_{size}.pdf'))
    plt.show()
    
compare_hurwitz(10000)
    


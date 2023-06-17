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
    concurrence0_ls = []
    concurrence1_ls = []
    concurrence2_ls = []
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
        concurrence0_ls.append(get_concurrence(rho0))
        concurrence1_ls.append(get_concurrence(rho1))
        concurrence2_ls.append(get_concurrence(rho2))

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

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), )
    ax[0].hist(purity0_ls, bins=20, alpha=.5, color='red', label='Method 0')
    ax[0].hist(purity1_ls, bins=20, alpha=.5, color='blue', label='Method 1')
    ax[0].hist(purity2_ls, bins=20, alpha=.5, color='green', label='Method 2')
    ax[0].set_title('Purity Distribution')
    ax[0].set_xlabel('Purity')
    ax[0].legend()
    ax[1].hist(concurrence0_ls, bins=20, alpha=.5, color='red', label='Method 0')
    ax[1].hist(concurrence1_ls, bins=20, alpha=.5, color='blue', label='Method 1')
    ax[1].hist(concurrence2_ls, bins=20, alpha=.5, color='green', label='Method 2')
    ax[1].set_title('Concurrence Distribution')
    ax[1].set_xlabel('Concurrence')
    ax[1].legend()
    ax[2].hist(phi0_ls, bins=20, alpha=.5, color='red', label='Method 0')
    ax[2].hist(phi1_ls, bins=20, alpha=.5, color='blue', label='Method 1')
    ax[2].hist(phi2_ls, bins=20, alpha=.5, color='green', label='Method 2')
    ax[2].set_title('Phi Distruibution')
    ax[2].legend()
    ax[2].set_xlabel('Angle (Rad)')
    plt.suptitle(f'Comparison of Hurwitz Method for {size} Random Density Matrices')
    plt.tight_layout()
    plt.savefig(join('random_compare', f'hurwitz_compare_{size}.pdf'))
    plt.show()

    print('mean purity method 0:', np.mean(purity0_ls))
    print('sem purity method 0:', np.std(purity0_ls)/np.sqrt(size))
    print('mean purity method 1:', np.mean(purity1_ls))
    print('sem purity method 1:', np.std(purity1_ls)/np.sqrt(size))
    print('mean purity method 2:', np.mean(purity2_ls))
    print('sem purity method 2:', np.std(purity2_ls)/np.sqrt(size))
    print('------------')
    print('mean concurrence method 0:', np.mean(concurrence0_ls))
    print('sem concurrence method 0:', np.std(concurrence0_ls)/np.sqrt(size))
    print('mean concurrence method 1:', np.mean(concurrence1_ls))
    print('sem concurrence method 1:', np.std(concurrence1_ls)/np.sqrt(size))
    print('mean concurrence method 2:', np.mean(concurrence2_ls))
    print('sem concurrence method 2:', np.std(concurrence2_ls)/np.sqrt(size))
    
size = int(input('Enter size of random sample: '))
compare_hurwitz(size)
    


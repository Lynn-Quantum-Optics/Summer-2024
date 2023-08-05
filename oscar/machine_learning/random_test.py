# file to get histograms of random density matrices
from rho_methods import *
from roik_gen import *
from tqdm import trange

def get_hist(N=10000):
    ''' get histogram of random density matrices: purity, concurrence, witness vals 
    --
    Params:
        N (int): number of random density matrices to generate
    '''
    # get rstats #
    purity_ls = []
    concurrence_ls = []
    W_ls = []
    Wp_ls = []
    for i in trange(N):
        rho = get_random_rho()
        purity_ls.append(get_purity(rho))
        concurrence_ls.append(get_concurrence(rho))
        witness = compute_witnesses(rho)
        W_ls.append(witness[0])
        Wp_ls.append(min(witness[1:]))
    # plot histograms #
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].hist(purity_ls, bins=20)
    ax[0, 0].set_title('Purity Distribution')
    ax[0, 1].hist(concurrence_ls, bins=20)
    ax[0, 1].set_title('Concurrence Distribution')
    ax[1, 0].hist(W_ls, bins=20)
    ax[1, 0].set_title('$W$ Distribution')
    ax[1, 1].hist(Wp_ls, bins=20)
    ax[1, 1].set_title('$W\'$ Distribution')
    plt.suptitle('Histograms of Random Roik Density Matrices')
    plt.tight_layout()
    plt.savefig(f'random_gen/data/{N}_hist.pdf')
    plt.show()

if __name__ == '__main__':
    get_hist()



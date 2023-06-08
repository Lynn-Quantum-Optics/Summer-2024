# file to check entanglement of a density matrix
# adapted from Alec's code and mine

# main imports #
from os.path import join
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from tqdm import trange

# random state gen imports #
from jones_simplex_datagen import get_random_jones, get_random_simplex
from o_r_datagen import get_random_roik

def get_min_eig(M0):
    '''
    Computes the eigenvalues of the partial transpose; if at least one is negative, then state labeled as '0' for entangled; else, '1'. 
    '''
    def partial_transpose(M0):
        # decompose M0 into blocks
        b1 = M0[:2, :2]
        b2 = M0[:2, 2:]
        b3 = M0[2:, :2]
        b4 = M0[2:, 2:]

        PT = np.matrix(np.block([[b1.T, b2.T], [b3.T, b4.T]]))
        return PT

    # compute partial tranpose
    PT = partial_transpose(M0)
    eigenvals = la.eigvals(PT)
    eigenvals.sort() # sort

    return np.real(eigenvals[0]) # return min eigenvalue

def get_concurrence(rho):
    ''' Calculates concurrence of a density matrix using R matrix. '''
    def R_matrix(rho):
        ''' Calculates the Hermitian R matrix for finding concurrence. '''
        def spin_flip(rho):
            ''' Returns the 'spin-flipped' (tilde) version of a density matrix rho.'''
            # define spin operators
            sy = np.array([[0,-1j],[1j,0]])
            sysy = np.kron(sy,sy)
            # perform spin flipping
            return sysy @ rho.conj() @ sysy
        sqrt_rho = la.sqrtm(rho)
        rho_tilde = spin_flip(rho)
        return (sqrt_rho @ rho_tilde @ sqrt_rho)
    R = R_matrix(rho)
    eig_vals = np.real(la.eigvals(R))
    eig_vals = np.sort(eig_vals)[::-1] # reverse sort numpy array
    return np.max([0,eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3]])

def check_entangled(rho, print=False):
    ''' Checks if a density matrix is entangled, using both concurrence and PPT. '''
    concurrence = get_concurrence(rho)
    min_eig = get_min_eig(rho)
    if print:
        print('Concurrence: ', concurrence)
        print('Min eigenvalue: ', min_eig)
    return concurrence, min_eig

def check_entangled_sample(N=10000, conditions=None, func=get_random_simplex, method_name='simplex', savedir='rho_test_plots', special_name='0'):
    ''' Checks random sample of N simplex generated matrices. 
    params:
        N: number of random states to check
        conditions: list of conditions to check for: tuple of tuple for min max bounds for concurrence and min eigenvalue. e.g. ((0, .5), (0, -.5)) will ensure states have concurrence between 0 and 0.5 and min eigenvalue between 0 and -0.5
        func: function to generate random state
        method_name: name of method used to generate random state
        savedir: directory to save plots
    '''
    concurrence_ls = []
    min_eig_ls = []
    # for n in trange(N):
    for n in range(N):
        # get state
        def get_state():
            if func.__name__ == 'get_random_jones' or func.__name__ == 'get_random_simplex':
                state = func()[0]
            else:
                state=func()
            return state
        # impose conditions
        if conditions != None:
            go=False
            while not(go):
                state = get_state()
                concurrence, min_eig = check_entangled(state)
                if conditions[0][0] <= concurrence <= conditions[0][1] and conditions[1][0] <= min_eig <= conditions[1][1]:
                    print(state)
                    print('is Hermitian', all(state==state.H))
                    go=True
                else:
                    concurrence, min_eig = check_entangled(get_state())
        else:
            # check if entangled
            concurrence, min_eig = check_entangled(get_state())
        # plot
        concurrence_ls.append(concurrence)
        min_eig_ls.append(min_eig)

    # fig, axes = plt.subplots(2,1, figsize=(10,5))
    # axes[0].hist(concurrence_ls, bins=100)
    # axes[1].hist(min_eig_ls, bins=100)
    # plt.show()
    plt.figure(figsize=(10,7))
    plt.plot(concurrence_ls, min_eig_ls, 'o')
    plt.xlabel('Concurrence')
    plt.ylabel('Min eigenvalue')
    plt.title('Concurrence vs. min eigenvalue for %s'%method_name)
    plt.savefig(join(savedir, 'concurrence_vs_min_eig_%i_%s_%s.pdf'%(N, method_name, special_name)))
    # plt.show()

# presets
if __name__ == '__main__':
    # no conditions
    # check_entangled_sample()
    # check_entangled_sample(func=get_random_roik, method_name='roik')
    # check_entangled_sample(func=get_random_jones, method_name='jones')

    # conditions
    check_entangled_sample(N=100, method_name='jones', conditions=((0, 0), (-1000, 0)), func=get_random_jones, special_name='conc_0')
    # check_entangled_sample(N=1000, method_name='roik', conditions=((0, 0), (-1000, 0)), func=get_random_roik, special_name='conc_0')
    pass
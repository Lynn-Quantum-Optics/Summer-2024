# file to generate random states and compare the W values with my old implementation to paco's implementation
from compute_witnesses_old import compute_witnesses_old
import os, sys
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import numpy as np


higher_directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(higher_directory_path)
from rho_methods_wp_final import compute_witnesses, get_concurrence, is_valid_rho, get_purity, adjoint

def get_random_hurwitz(method=1, log_params=False, conc_cond = 0, purity_cond = 1):
    ''' Function to generate random density matrix with roik method.
    params:
        method: int, 0, 1, 2 for whether to use 
            (0) phi = arcsin(xi^1/n) for n in 1 to 6 with xi : [0,1) or
            (1) phi = arcsin(xi^1/2) for n in 1 to 6 or
            (2) phi = rand in [0, pi/2] for n in 1 to 6
            default is 0.
        log_params: bool, whether to return the parameters used to generate the 3rd unitary matrix
        conc_cond: default is 0, will generate random density matrix until concurrence is >= than this value
        purity_condition: default is 1; will generate random density matrix until purity is less than this value
    '''
    ## part 1: random diagonal elements ##
    def rand_diag():
        # get 4 random params
        
        M11 = np.random.randint(0, 100000) / 100000
        M22 = np.random.randint(0, 100000) / 100000*(1-M11)
        M33 = np.random.randint(0, 100000) / 100000*(1-M11 - M22)
        M44 = 1-M11-M22-M33
        
        # shuffle the entries
        rand_elems = np.array([M11, M22, M33, M44])
        np.random.shuffle(rand_elems)
        M = np.matrix(np.diag([rand_elems[0], rand_elems[1], rand_elems[2], rand_elems[3]]))
        return M

    ## part 2: random unitary trans ##
    global params
    def rand_unitary():
        # need to first generate 6 other smaller unitaries
        if method==1 or method==2:
            def get_rand_elems():
                alpha = np.random.randint(0, 200000*np.pi) / 100000
                psi = np.random.randint(0, 200000*np.pi) / 100000
                chi = np.random.randint(0, 200000*np.pi) / 100000
                if method==1:
                    phi = np.arcsin((np.random.randint(0, 100000)/100000)**(1/2))
                else: # method==2
                    phi = np.random.rand()*np.pi/2
                return np.matrix([
                    [np.e**(psi*1j)*np.cos(phi), np.e**(chi*1j)*np.sin(phi)],
                    [-np.e**(-chi*1j)*np.sin(phi), np.e**(-psi*1j)*np.cos(phi)]
                ])*np.e**(alpha*1j), alpha, psi, chi, phi

            # loop and create unitaries from blocks
            unitary_final = np.eye(4, dtype=np.complex128)
            for k in range(5, -1, -1): # count down to do multiplicatiom from right to left
                sub_unitary_k,   alpha, psi, chi, phi = get_rand_elems()
                if k==0 or k==3 or k==5:
                    unitary_k = np.matrix(np.block([[np.eye(2), np.zeros((2,2))], [np.zeros((2,2)), sub_unitary_k]]))
                    if log_params and k==3:
                        global params
                        params = [alpha, psi, chi, phi]
                elif k==1 or k==4:
                    ul = np.matrix([[1,0], [0, sub_unitary_k[0, 0]]])# upper left
                    ur = np.matrix([[0,0], [sub_unitary_k[0, 1], 0]])# upper right
                    ll = np.matrix([[0,sub_unitary_k[1,0]], [0, 0]])# lower left
                    lr = np.matrix([[sub_unitary_k[1,1], 0], [0, 1]])# lower right
                    unitary_k = np.matrix(np.block([[ul, ur],[ll, lr]]))
                else: # k==2
                    unitary_k = np.matrix(np.block([[sub_unitary_k, np.zeros((2,2))], [np.zeros((2,2)), np.eye(2)]]))
                
                # print(np.all(np.isclose(unitary_k@np.linalg.inv(unitary_k), np.eye(4), atol=1e-9)))
                
                unitary_final =  unitary_k @ unitary_final# this way correctly builds right to left
        else: # method==0
            def get_U(i, j, k):
                # get i phi and psi
                phi = np.arcsin((np.random.rand(i+1))**(1/(i+1)))
                psi = np.random.rand(i+1)*2*np.pi
                # get 1 random chi
                if k==0:
                    chi = 0
                else:
                    chi = np.random.rand()*2*np.pi

                # start with identity
                U_k = np.eye(4, dtype=np.complex128)
                # fill in the entries
                U_k[i][i]=np.e**(psi[i]*1j)*np.cos(phi[i])
                U_k[i][j]=np.e**(chi*1j)*np.sin(phi[i])
                U_k[j][i]=-np.e**(-chi*1j)*np.sin(phi[i])
                U_k[j][j]=np.e**(-psi[i]*1j)*np.cos(phi[i])

                return U_k

            unitary_final = np.eye(4, dtype=np.complex128)
            for l in range(2, -1, -1): # outer loop for the 3 unitaries
                U_l = np.eye(4)
                for k in range(l, -1, -1): # build the unitary from right to left
                    U_l = U_l @ get_U(k, l+1, l)
                unitary_final = unitary_final @ U_l
            
            alpha = np.random.rand()*2*np.pi
            unitary_final = unitary_final * np.e**(alpha*1j)

        return unitary_final

     ## combine M and U as follows: U M U^\dagger
    def combine_rand(): 
        return U @ M @ adjoint(U) # calling adjoint method from rho_methods

    M = rand_diag()
    U = rand_unitary()
    M0 = combine_rand()

    while not(is_valid_rho(M0)) or (get_purity(M0)>purity_cond and get_concurrence(M) >= conc_cond): # if not valid density matrix, keep generating
        print('invalid!!!')
        print(M0)
        M = rand_diag()
        U = rand_unitary()
        M0 = combine_rand()

    # return M0, get_purity(M0)
    if not(log_params): return M0
    else: return [M0, params]

def evaluate(_):
    rho = get_random_hurwitz()
    concurrence = get_concurrence(rho)
    w_old, wp1_old, wp2_old, wp3_old = compute_witnesses_old(rho)
    w_new, wp1_new, wp2_new, wp3_new = compute_witnesses(rho)

    return {'w_old': w_old, 'wp1_old': wp1_old, 'wp2_old': wp2_old, 'wp3_old': wp3_old, 'w_new': w_new, 'wp1_new': wp1_new, 'wp2_new': wp2_new, 'wp3_new': wp3_new, 'concurrence': concurrence}

def parallel_evaluate(n_iterations):
    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(evaluate, range(n_iterations))
    t1 = time.time()
    print(f'Parallel evaluation took {t1-t0} seconds')
    results = pd.DataFrame(results)
    results.to_csv(f'witness_comparison_{n_iterations}.csv', index=False)
    
    return results

def plot_result(n_iterations):
    file = f'witness_comparison_{n_iterations}.csv'
    results = pd.read_csv(file)
    w_old = results['w_old']
    w_new = results['w_new']
    wp1_old = results['wp1_old']
    wp1_new = results['wp1_new']
    wp2_old = results['wp2_old']
    wp2_new = results['wp2_new']
    wp3_old = results['wp3_old']
    wp3_new = results['wp3_new']

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].scatter(w_old, w_new)
    axs[0, 0].set_xlabel('Old W')
    axs[0, 0].set_ylabel('New W')
    axs[0, 0].set_title('W')
    axs[0, 1].scatter(wp1_old, wp1_new)
    axs[0, 1].set_xlabel('Old WP1')
    axs[0, 1].set_ylabel('New WP1')
    axs[0, 1].set_title('WP1')
    axs[1, 0].scatter(wp2_old, wp2_new)
    axs[1, 0].set_xlabel('Old WP2')
    axs[1, 0].set_ylabel('New WP2')
    axs[1, 0].set_title('WP2')
    axs[1, 1].scatter(wp3_old, wp3_new)
    axs[1, 1].set_xlabel('Old WP3')
    axs[1, 1].set_ylabel('New WP3')
    axs[1, 1].set_title('WP3')

    # draw straight line thru origin
    for ax in axs.flat:
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    # calculate R^2 for each plot
    def r(x, y):
        return np.corrcoef(x, y)[0, 1]
    
    r_w = r(w_old, w_new)
    r_wp1 = r(wp1_old, wp1_new)
    r_wp2 = r(wp2_old, wp2_new)
    r_wp3 = r(wp3_old, wp3_new)

    axs[0, 0].text(0.5, 0.1, f'$R = {r_w}$', horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.5, 0.1, f'$R = {r_wp1}$', horizontalalignment='center', verticalalignment='center', transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.5, 0.1, f'$R = {r_wp2}$', horizontalalignment='center', verticalalignment='center', transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.5, 0.1, f'$R = {r_wp3}$', horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)

    plt.savefig(f'witness_comparison_{n_iterations}.pdf')
    plt.show()
    

if __name__ == '__main__':
    n_iterations = 60
    parallel_evaluate(n_iterations)
    plot_result(n_iterations)


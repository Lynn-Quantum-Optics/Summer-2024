# file based on generator_general_state.py to create datasets for training\
import numpy as np
import pandas as pd
from os.path import join
from tqdm import trange # for progress bar

# import for rho_test ##
from rho_methods import is_valid_rho, get_purity, get_min_eig, get_all_roik_projections

### randomization functions ###
def get_random_roik():
    ## method 1: random diagonal elements ##
    def rand_diag():
        # get 4 random params
        def get_rand_elems():
            M11 = np.random.rand()
            M22 = np.random.rand()*(1-M11)
            M33 = np.random.rand()*(1-M11 - M22)
            M44 = np.random.rand()*(1-M11-M22-M33)
            
            # shuffle the entries
            rand_elems = np.array([M11, M22, M33, M44])
            np.random.shuffle(rand_elems)
            return M11+M22+M33+M44, rand_elems[0], rand_elems[1], rand_elems[2], rand_elems[3]
        N = 0 # set to 0 initially
        while N ==0:
            N, M11, M22, M33, M44 = get_rand_elems()
            # define new 4x4 matrix; cast the diagonal array as matrix for more convienient fnuctionality
        M = np.matrix(np.diag([M11/N, M22/N, M33/N, M44/N]))
        return M

    ## method 2: random unitary trans ##
    def rand_unitary():
        # need to first generate 6 other smaller unitaries
        tau = np.pi*2
        e = np.e
        ra = np.pi/2
        def get_rand_elems():
            alpha = np.random.rand()*tau
            psi = np.random.rand()*tau
            chi = np.random.rand()*tau
            phi = np.random.rand()*ra

            return np.matrix([
                [e**(psi*1j)*np.cos(phi), e**(chi*1j)*np.sin(phi)],
                [-e**(-chi*1j)*np.sin(phi), e**(-psi*1j)*np.cos(phi)]
            ])*e**(alpha*1j)

        # loop and create unitaries from blocks
        unitary_final = np.eye(4)
        for k in range(5, -1, -1): # count down to do multiplicatiom from right to left
            sub_unitary_k = get_rand_elems()
            if k==0 or k==3 or k==5:
                unitary_k = np.matrix(np.block([[np.eye(2), np.zeros((2,2))], [np.zeros((2,2)), sub_unitary_k]]))
            elif k==1 or k==4:
                ul = np.matrix([[1,0], [0, sub_unitary_k[0, 0]]])# upper left
                ur = np.matrix([[0,0], [sub_unitary_k[0, 1], 0]])# upper right
                ll = np.matrix([[0,sub_unitary_k[1,0]], [0, 0]])# lower left
                lr = np.matrix([[sub_unitary_k[1,1], 0], [0, 1]])# lower right
                unitary_k = np.matrix(np.block([[ul, ur],[ll, lr]]))
            else: # k==2
                unitary_k = np.matrix(np.block([[sub_unitary_k, np.zeros((2,2))], [np.zeros((2,2)), np.eye(2)]]))
            
            unitary_final = unitary_final @ unitary_k

        return unitary_final

    ## combine M and U as follows: U M U^\dagger
    def combine_rand(): 
        return U @ M @ U.H # .H does conjugate transpose

    M = rand_diag()
    U = rand_unitary()
    M0 = combine_rand(M, U)

    while not(is_valid_rho(M0)): # if not valid density matrix, keep generating
        M = rand_diag()
        U = rand_unitary()
        M0 = combine_rand(M, U)

    return M0

## generate the complete dataset ##
def gen_dataset(size, savepath):
    '''
    Takes as input the length of the desired dataset and returns a csv of randomized states as well as path to save
    '''

    M0 = get_random_roik() # get the randomized state

    HH, VV, HV, DD, AA, RR, LL, DL, AR, DH, AV, LH, RV, DR, DV, LV = get_all_roik_projections(M0)

    # initialize dataframes
    df_3 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'min_eig':[], 'purity':[]})
    df_5 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'DD':[], 'AA':[], 'min_eig':[], 'purity':[]})
    df_6 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'DD':[], 'RR':[], 'LL':[], 'min_eig':[], 'purity':[]})
    df_12 = pd.DataFrame({'DD':[], 'AA':[], 'DL':[], 'AR':[], 'DH':[], 'AV':[], 'LL':[], 'RR':[], 'LH':[], 'RV':[], 'HH':[], 'VV':[], 'min_eig':[], 'purity':[]})
    df_15 = pd.DataFrame({'DD':[], 'AA':[], 'DL':[], 'AR':[], 'DH':[], 'AV':[], 'LL':[], 'RR':[], 'LH':[], 'RV':[], 'HH':[], 'VV':[], 'DR':[], 'DV':[], 'LV':[], 'min_eig':[], 'purity':[]})
    
    for j in trange(size):
        # get the randomized state
        M0 = get_random_roik() 
        min_eig = get_min_eig(M0)
        purity = get_purity(M0)

        # compute projections in groups
        df_3 = pd.concat([df_3, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'min_eig':min_eig,'purity':purity}])])        
        df_5 = pd.concat([df_5, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'DD':DD, 'AA':AA, 'min_eig':min_eig,'purity':purity}])])
        df_6 = pd.concat([df_6, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'DD':DD, 'RR':RR,'LL': LL,'min_eig':min_eig,'purity':purity}])])
        df_12 = pd.concat([df_12, pd.DataFrame.from_records([{'DD':DD, 'AA':AA, 'DL':DL, 'AR':AR, 'DH':DH, 'AV':AV, 'LL':LL, 'RR':RR, 'LH':LH, 'RV':RV, 'HH':HH, 'VV':VV, 'min_eig':min_eig,'purity':purity}])])
        df_15 = pd.concat([df_15, pd.DataFrame.from_records([{'DD':DD, 'AA':AA, 'DL':DL, 'AR':AR, 'DH':DH, 'AV':AV, 'LL':LL, 'RR':RR, 'LH':LH, 'RV':RV, 'HH':HH, 'VV':VV, 'DR':DR, 'DV':DV, 'LV':LV, 'min_eig':min_eig,'purity':purity}])])

    df_3.to_csv(join(savepath, 'df_3.csv'))
    df_5.to_csv(join(savepath, 'df_5.csv'))
    df_6.to_csv(join(savepath, 'df_6.csv'))
    df_12.to_csv(join(savepath, 'df_12.csv'))
    df_15.to_csv(join(savepath, 'df_15.csv'))

## build dataset ##
if __name__ == '__main__':
    size=44
    savepath='RO_data'
    gen_dataset(size, savepath)

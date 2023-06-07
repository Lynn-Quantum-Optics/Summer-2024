# file based on generator_general_state.py to create datasets for training\
import numpy as np
import pandas as pd
from os.path import join
from tqdm import trange # for progress bar

### randomization functions ###
def get_roik_random():
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
    def combine_rand(M, U): 
        return U @ M @ U.H # .H does conjugate transpose

    M = rand_diag()
    U = rand_unitary()
    M0 = combine_rand(M, U)

    return M0

## generate the complete dataset ##
def gen_dataset(size, savepath):
    '''
    Takes as input the length of the desired dataset and returns a csv of randomized states as well as path to save
    '''

    def compute_proj(basis1, basis2, M0):
        '''
        Computes projection into desired bases.
        '''
        Bell_singlet = np.matrix([[0, 0, 0, 0], [0, .5, -.5, 0], [0, -0.5, .5, 0], [0, 0, 0, 0]])

        M0_swapped = M0.copy() # swap the subsystems A and B
        M0_swapped[:, 1] = M0[:, 2]
        M0_swapped[:, 2] = M0[:, 1]

        M_T = np.kron(M0, M0_swapped)
        num = M_T @ np.kron(np.kron(basis1, Bell_singlet), basis2)
        denom = M_T @ np.kron(np.kron(basis1, np.eye(4)), basis2)

        try: # compute the projection as defined in Roik et al
            return (np.trace(num) / np.trace(denom)).real
        except ZeroDivisionError:
            return 0

    def get_purity(M0):
        return (np.trace(np.linalg.matrix_power(M0, 2))).real # returns trace of reduced density matrix ^2

    def check_entangled(M0):
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
        eigenvals = np.linalg.eigvals(PT)
        eigenvals.sort() # sort

        # output = 1
        # if eigenvals[0] < 0: # min neg eigenval implies entangled; else, stays 1 for separable
        #     output=0

        return eigenvals[0].real # return min eigenvalue

    # define the single bases for projection
    H = np.array([[1,0],[0,0]])
    V = np.array([[0,0],[0,1]])    
    D = np.array([[1/2,1/2],[1/2,1/2]])
    A = np.array([[1/2,-1/2],[-1/2,1/2]])
    R = np.array([[1/2,1j/2],[-1j/2,1/2]])
    L = np.array([[1/2,-1j/2],[1j/2,1/2]])

    # initialize dataframes
    df_3 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'min_eig':[], 'purity':[]})
    df_5 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'DD':[], 'AA':[], 'min_eig':[], 'purity':[]})
    df_6 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'DD':[], 'RR':[], 'LL':[], 'min_eig':[], 'purity':[]})
    df_12 = pd.DataFrame({'DD':[], 'AA':[], 'DL':[], 'AR':[], 'DH':[], 'AV':[], 'LL':[], 'RR':[], 'LH':[], 'RV':[], 'HH':[], 'VV':[], 'min_eig':[], 'purity':[]})
    df_15 = pd.DataFrame({'DD':[], 'AA':[], 'DL':[], 'AR':[], 'DH':[], 'AV':[], 'LL':[], 'RR':[], 'LH':[], 'RV':[], 'HH':[], 'VV':[], 'DR':[], 'DV':[], 'LV':[], 'min_eig':[], 'purity':[]})
    
    for j in trange(size):
        # get the randomized state
        M0 = get_roik_random() 
        min_eig = check_entangled(M0)
        purity = get_purity(M0)

        # compute projections in groups
        HH = compute_proj(H, H, M0)
        VV = compute_proj(V, V, M0)
        HV = compute_proj(H, V, M0)
        df_3 = pd.concat([df_3, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'min_eig':min_eig,'purity':purity}])])

        DD = compute_proj(D, D, M0)
        AA = compute_proj(A, A, M0)
        df_5 = pd.concat([df_5, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'DD':DD, 'AA':AA, 'min_eig':min_eig,'purity':purity}])])

        RR = compute_proj(R,R, M0)
        LL = compute_proj(L, L, M0)
        df_6 = pd.concat([df_6, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'DD':DD, 'RR':RR,'LL': LL,'min_eig':min_eig,'purity':purity}])])

        DL = compute_proj(D,L, M0)
        AR = compute_proj(A,R, M0)
        DH = compute_proj(D,H, M0)
        AV = compute_proj(A,V, M0)
        LH = compute_proj(L,H, M0)
        RV = compute_proj(R,V, M0)
        df_12 = pd.concat([df_12, pd.DataFrame.from_records([{'DD':DD, 'AA':AA, 'DL':DL, 'AR':AR, 'DH':DH, 'AV':AV, 'LL':LL, 'RR':RR, 'LH':LH, 'RV':RV, 'HH':HH, 'VV':VV, 'min_eig':min_eig,'purity':purity}])])

        DR = compute_proj(D,R, M0)
        DV = compute_proj(D,V, M0)
        LV = compute_proj(L,V, M0)
        df_15 = pd.concat([df_15, pd.DataFrame.from_records([{'DD':DD, 'AA':AA, 'DL':DL, 'AR':AR, 'DH':DH, 'AV':AV, 'LL':LL, 'RR':RR, 'LH':LH, 'RV':RV, 'HH':HH, 'VV':VV, 'DR':DR, 'DV':DV, 'LV':LV, 'min_eig':min_eig,'purity':purity}])])

    df_3.to_csv(join(savepath, 'df_3.csv'))
    df_5.to_csv(join(savepath, 'df_5.csv'))
    df_6.to_csv(join(savepath, 'df_6.csv'))
    df_12.to_csv(join(savepath, 'df_12.csv'))
    df_15.to_csv(join(savepath, 'df_15.csv'))

if __name__ == '__main__':
    size=4400000
    savepath='RO_data'
    gen_dataset(size, savepath)

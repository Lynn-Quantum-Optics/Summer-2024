# file to use the jones matrices to randomly generate entangled states
from os.path import join
import numpy as np
import pandas as pd
from os.path import join
import scipy.linalg as la
from tqdm import trange # for progress bar

from rho_methods import is_valid_rho, get_min_eig, compute_witnesses, get_all_projections
from jones import get_Jrho_C

def get_random_jones():
    ''' Computes random angles in the ranges specified and generates the resulting states'''
    def get_random_angles_C():
        ''' Returns random angles for the Jrho_C setup'''
        theta_ls = np.random.rand(2)*np.pi/4
        theta1, theta2 = theta_ls[0], theta_ls[1]
        alpha_ls = np.random.rand(2)*np.pi/2
        alpha1, alpha2 = alpha_ls[0], alpha_ls[1]
        phi = np.random.rand()*0.69 # experimental limit of our QP

        return [theta1, theta2, alpha1, alpha2, phi]

    angles = get_random_angles_C()
    rho = get_Jrho_C(*angles)

    # call method to confirm state is valid
    while not(is_valid_rho(rho)):
        angles = get_random_angles_C()
        rho = get_Jrho_C(*angles)

    return [rho, angles]

def get_random_simplex():
    '''
    Returns density matrix for random state of form:
    a|HH> + be^(i*beta)|01> + ce^(i*gamma)*|10> + de^(i*delta)*|11>
    '''
    
    a = np.random.rand()
    b = np.random.rand() *(1-a)
    c = np.random.rand()*(1-a-b)
    d = 1-a-b-c

    real_ls = [] # list to store real coefficients
    real_ls = np.sqrt(np.array([a, b, c, d]))
    np.random.shuffle(real_ls)
    rand_angle=np.random.rand(3)*2*np.pi

    state_vec = np.multiply(real_ls, np.e**(np.concatenate((np.array([1]), rand_angle))*1j)).reshape((4,1))

    # compute density matrix
    rho = np.matrix(state_vec @ np.conjugate(state_vec.reshape((1,4))))

    return [rho, np.concatenate((real_ls, rand_angle))]

def analyze_state(rho_angles, rand_type):
    '''
    Takes as input a density matrix, outputs dictionary with angles, projection probabilities, W values, and W' values. 
    rand_type is a string, either <jones> or <simplex>, which builds the correct df
    '''
    rho = rho_angles[0]
    angles = rho_angles[1]
    
    HH, HV, VH, VV, DD, DA, AD, AA, RR, RL, LR, LL = get_all_projections(rho)

    ## compute W and W' ##
    W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho)

    min_eig = get_min_eig(rho)
   
    if rand_type=='jones':
        return {'theta1':angles[0], 'theta2':angles[1], 'alpha1':angles[2], 'alpha2':angles[3], 'phi':angles[4],
        'HH':HH, 'HV':HV,'VH':VH, 'VV':VV, 'DD':DD, 'DA':DA, 'AD':AD, 'AA':AA, 
        'RR':RR, 'RL':RL, 'LR':LR, 'LL':LL, 'W_min':W_min, 'Wp_t1': Wp_t1,'Wp_t2': Wp_t2, 'Wp_t3': Wp_t3, 'min_eig':min_eig}
    elif rand_type =='simplex':
        return {'a':angles[0], 'b':angles[1], 'c':angles[2], 'd':angles[3], 'beta':angles[4], 'gamma':angles[5], 'delta':angles[6],
        'HH':HH, 'HV':HV,'VH':VH, 'VV':VV, 'DD':DD, 'DA':DA, 'AD':AD, 'AA':AA, 
        'RR':RR, 'RL':RL, 'LR':LR, 'LL':LL, 'W_min':W_min, 'Wp_t1': Wp_t1,'Wp_t2': Wp_t2, 'Wp_t3': Wp_t3, 'min_eig':min_eig}
    else:
        print('Incorrect rand_type.')

## perform randomization ##
def gen_data(N=50000, do_jones=True, do_simplex=True, DATA_PATH='jones_simplex_data'):
    '''
    Generates random states and computes 12 input probabilities, W min, W' min for each triplet, and min_eig for each state.
    params:
        N: number of states to generate
        do_jones: if True, generate states using Jones generation
        do_simplex: if True, generate states using simplex generation
        DATA_PATH: path to save data
    '''
    if do_jones and do_simplex:
        # initilize dataframe to hold states
        df_jones = pd.DataFrame({'theta1':[], 'theta2':[], 'alpha1':[], 'alpha2':[], 'phi':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        df_simplex = pd.DataFrame({'a':[], 'b':[], 'c':[], 'd':[], 'beta':[], 'gamma':[], 'delta':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        for i in trange(N):
            df_jones = pd.concat([df_jones, pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])])
            df_simplex = pd.concat([df_simplex, pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])])
        # save!
        df_jones.to_csv(join(DATA_PATH, 'jones_%i_0.csv'%N))
        df_simplex.to_csv(join(DATA_PATH, 'simplex_%i_0.csv'%N))
    elif do_jones:
        df_jones = pd.DataFrame({'theta1':[], 'theta2':[], 'alpha1':[], 'alpha2':[], 'phi':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        for i in trange(N):
            df_jones = pd.concat([df_jones, pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])])
        # save!
        df_jones.to_csv(join(DATA_PATH, 'jones_%i_0.csv'%N))
    elif do_simplex:
        df_simplex = pd.DataFrame({'a':[], 'b':[], 'c':[], 'd':[], 'beta':[], 'gamma':[], 'delta':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        for i in trange(N):
            df_simplex = pd.concat([df_simplex, pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])])
        # save!
        df_simplex.to_csv(join(DATA_PATH, 'simplex_%i_0.csv'%N))
    else:
        print('Hmmm... make sure one of do_jones or do_simplex is enabled.')
    

if __name__=='__main__':
    gen_data(do_jones=True, do_simplex=False)
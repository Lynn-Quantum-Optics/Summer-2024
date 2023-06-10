# file to use the jones matrices to randomly generate entangled states
from os.path import join
import numpy as np
import pandas as pd
from os.path import join
import scipy.linalg as la
from tqdm import trange # for progress bar

from rho_methods import is_valid_rho, get_min_eig, compute_witnesses, get_all_projections
from jones import get_random_jones
from random_gen import get_random_simplex


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
def gen_data(N=50000, do_jones=True, do_simplex=True, DATA_PATH='jones_simplex_data', special='0', restrict=False):
    '''
    Generates random states and computes 12 input probabilities, W min, W' min for each triplet, and min_eig for each state.
    params:
        N: number of states to generate
        do_jones: if True, generate states using Jones generation
        do_simplex: if True, generate states using simplex generation
        DATA_PATH: path to save data
        special: special identifier for file
    '''
    if do_jones and do_simplex:
        # initilize dataframe to hold states
        df_jones = pd.DataFrame({'theta1':[], 'theta2':[], 'alpha1':[], 'alpha2':[], 'phi':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        df_simplex = pd.DataFrame({'a':[], 'b':[], 'c':[], 'd':[], 'beta':[], 'gamma':[], 'delta':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        if not(restrict):
            for i in trange(N):
                df_jones = pd.concat([df_jones, pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])])
                df_simplex = pd.concat([df_simplex, pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])])
        else:
            for i in trange(N):
                jones_new =pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])
                simplex_new = pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])

                # impose condition
                while not(jones_new['W_min'].values[0]>=0 and (jones_new['Wp_t1'].values[0]<0 or jones_new['Wp_t2'].values[0]<0 or jones_new['Wp_t3'].values[0]<0)):
                    jones_new =pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])
                while not(simplex_new['W_min'].values[0]>=0 and (simplex_new['Wp_t1'].values[0]<0 or simplex_new['Wp_t2'].values[0]<0 or simplex_new['Wp_t3'].values[0]<0)):
                    simplex_new = pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])

                df_jones = pd.concat([df_jones, jones_new])
                df_simplex = pd.concat([df_simplex, simplex_new])

        # save!
        df_jones.to_csv(join(DATA_PATH, 'jones_%i_%s.csv'%(N, special)))
        df_simplex.to_csv(join(DATA_PATH, 'simplex_%i_%s.csv'%(N, special)))
    elif do_jones:
        df_jones = pd.DataFrame({'theta1':[], 'theta2':[], 'alpha1':[], 'alpha2':[], 'phi':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        if not(restrict):
            for i in trange(N):
                df_jones = pd.concat([df_jones, pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])])
        else:
            jones_new =pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])
            
            # impose condition
            while not(jones_new['W_min'].values[0]>=0 and (jones_new['Wp_t1'].values[0]<0 or jones_new['Wp_t2'].values[0]<0 or jones_new['Wp_t3'].values[0]<0)):
                jones_new =pd.DataFrame.from_records([analyze_state(get_random_jones(), 'jones')])

            df_jones = pd.concat([df_jones, jones_new])
        # save!
        df_jones.to_csv(join(DATA_PATH, 'jones_%i_%s.csv'%(N, special)))
    elif do_simplex:
        df_simplex = pd.DataFrame({'a':[], 'b':[], 'c':[], 'd':[], 'beta':[], 'gamma':[], 'delta':[],
            'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
            'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
        if not(restrict):
            for i in trange(N):
                df_simplex = pd.concat([df_simplex, pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])])
        else:
            simplex_new = pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])

            # impose condition
            while not(simplex_new['W_min'].values[0]>=0 and (simplex_new['Wp_t1'].values[0]<0 or simplex_new['Wp_t2'].values[0]<0 or simplex_new['Wp_t3'].values[0]<0)): 
                simplex_new = pd.DataFrame.from_records([analyze_state(get_random_simplex(), 'simplex')])
            
            df_simplex = pd.concat([df_simplex, simplex_new])

        # save!
        df_simplex.to_csv(join(DATA_PATH, 'simplex_%i_%s.csv'%(N, special)))
    else:
        print('Hmmm... make sure one of do_jones or do_simplex is enabled.')
    

if __name__=='__main__':
    N = int(input('How many states to generate?'))
    do_j = int(input('Do Jones? (0/1)'))
    do_s = int(input('Do Simplex? (0/1)'))
    special = input('Special identifier for file?')
    restrict = int(input("Restrict to W>=0 but >= one of W' < 0? (0/1)"))
    print(bool(do_j), bool(do_s), bool(restrict))
    gen_data(N=N, do_jones=bool(do_j), do_simplex=bool(do_s), special=special, restrict=bool(restrict))
    # gen_data(N=100000, do_jones=False, do_simplex=True)
    # # gen_data(N=20000, do_jones=True, do_simplex=False)
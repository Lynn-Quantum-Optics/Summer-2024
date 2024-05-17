# file based on generator_general_state.py to create datasets for training\
import numpy as np
import pandas as pd
from os.path import join
from tqdm import trange # for progress bar

# import for rho_test ##
from rho_methods import get_purity, get_min_eig, get_all_roik_projections
from random_gen import get_random_roik

## generate the complete dataset ##
def gen_dataset(size, savepath, special):
    '''
    Takes as input the length of the desired dataset and returns a csv of randomized states as well as path to save
    special: special name identifier for the files
    '''

    # initialize dataframes
    df_3 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'min_eig':[], 'purity':[]})
    df_5 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'DD':[], 'AA':[], 'min_eig':[], 'purity':[]})
    df_6 = pd.DataFrame({'HH':[], 'VV':[], 'HV':[], 'DD':[], 'RR':[], 'LL':[], 'min_eig':[], 'purity':[]})
    df_12 = pd.DataFrame({'DD':[], 'AA':[], 'DL':[], 'AR':[], 'DH':[], 'AV':[], 'LL':[], 'RR':[], 'LH':[], 'RV':[], 'HH':[], 'VV':[], 'min_eig':[], 'purity':[]})
    df_15 = pd.DataFrame({'DD':[], 'AA':[], 'DL':[], 'AR':[], 'DH':[], 'AV':[], 'LL':[], 'RR':[], 'LH':[], 'RV':[], 'HH':[], 'VV':[], 'DR':[], 'DV':[], 'LV':[], 'min_eig':[], 'purity':[]})
    
    # for j in trange(size):
    for j in trange(size):
        # get the randomized state and important properties
        M0 = get_random_roik() 
        min_eig = get_min_eig(M0)
        purity = get_purity(M0)
        HH, VV, HV, DD, AA, RR, LL, DL, AR, DH, AV, LH, RV, DR, DV, LV = get_all_roik_projections(M0)
        # compute projections in groups
        df_3 = pd.concat([df_3, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'min_eig':min_eig,'purity':purity}])])        
        df_5 = pd.concat([df_5, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'DD':DD, 'AA':AA, 'min_eig':min_eig,'purity':purity}])])
        df_6 = pd.concat([df_6, pd.DataFrame.from_records([{'HH':HH, 'VV':VV, 'HV':HV,'DD':DD, 'RR':RR,'LL': LL,'min_eig':min_eig,'purity':purity}])])
        df_12 = pd.concat([df_12, pd.DataFrame.from_records([{'DD':DD, 'AA':AA, 'DL':DL, 'AR':AR, 'DH':DH, 'AV':AV, 'LL':LL, 'RR':RR, 'LH':LH, 'RV':RV, 'HH':HH, 'VV':VV, 'min_eig':min_eig,'purity':purity}])])
        df_15 = pd.concat([df_15, pd.DataFrame.from_records([{'DD':DD, 'AA':AA, 'DL':DL, 'AR':AR, 'DH':DH, 'AV':AV, 'LL':LL, 'RR':RR, 'LH':LH, 'RV':RV, 'HH':HH, 'VV':VV, 'DR':DR, 'DV':DV, 'LV':LV, 'min_eig':min_eig,'purity':purity}])])

    df_3.to_csv(join(savepath, 'df_3_%s.csv'%special))
    df_5.to_csv(join(savepath, 'df_5_%s.csv'%special))
    df_6.to_csv(join(savepath, 'df_6_%s.csv'%special))
    df_12.to_csv(join(savepath, 'df_12_%s.csv'%special))
    df_15.to_csv(join(savepath, 'df_15_%s.csv'%special))

## build dataset ##
if __name__ == '__main__':
    N = int(input('How many states to generate?'))
    special = input('Special name identifier for the files?')
    savepath='RO_data'
    gen_dataset(N, savepath, special)

# file to prepare data for ml
import os
from os.path import join
import pandas as pd
import numpy as np

def prepare_data(datapath, file, input_method, pop_method, task, split=True, p=0.8, normalize=False, conc_threshold=0, w_cond=True):
    ''' Function to prepare data for training.
    params:
        datapath: path to csv
        savename: what to name the prepped data
        input_method: how to prepare the input data: e.g., diag for XX, YY, ZZ
        pop_method: whether to include pop data inputs: raw, diff, rd, or none
        task: what task to train on: e.g., witness, entangled
        split: boolean for whether to split data into train and test
        p: fraction of data to use for training
        normalize: boolean for whether to normalize data column wise. DON'T USE.
        conc_threshold: threshold for concurrence to be considered entangled
        w_cond: boolean for whether to only include states that satisfy W condition; false to get complete dataset
    '''
    # print(join(datapath, file))
    # print('split', split)
    df= pd.read_csv(join(datapath, file))
    # print(df.head())

    if input_method == 'prob_3':
        inputs = ['HH', 'VV', 'HV']
    elif input_method=='prob_5':
        inputs = ['HH', 'HV', 'VV', 'DD', 'AA']
    elif input_method=='prob_6':
        inputs = ['HH', 'HV', 'VV', 'DD', 'RR', 'LL']
    elif input_method=='prob_9': 
        inputs = ['HH', 'HV', 'VV', 'DD', 'DA', 'AA', 'RR', 'RL', 'LL']
    elif input_method=='prob_12_red': # redudant prob based on including full bases for each
        inputs = ['HH', 'HV', 'VH', 'VV', 'DD', 'DA', 'AD', 'AA', 'RR', 'RL', 'LR', 'LL']
    elif input_method=='prob_12':
        inputs = ['DD', 'AA', 'DL', 'AR', 'DH', 'AV', 'LL', 'RR', 'LH', 'RV', 'HH', 'VV']
    elif input_method=='prob_15':
        inputs = ['DD', 'AA', 'DL', 'AR', 'DH', 'AV', 'LL', 'RR', 'LH', 'RV', 'HH', 'VV', 'DR', 'DV', 'LV']
    
    # roik specific probs #
    elif input_method=='prob_3_r':
        inputs = ['r_HH', 'r_VV', 'r_HV']
    elif input_method=='prob_5_r':
        inputs = ['r_HH', 'r_VV', 'r_HV', 'r_DD', 'r_AA']
    elif input_method=='prob_6_r':
        inputs = ['r_HH', 'r_VV', 'r_HV', 'r_DD', 'r_RR', 'r_LL']
    # elif input_method=='prob_9_r':
    #     inputs = ['r_HH', 'r_HV', 'r_VV', 'r_DD', 'r_DA', 'r_AA', 'r_RR', 'r_RL', 'r_LL']
    elif input_method=='prob_12_r':
        inputs = ['r_DD', 'r_AA', 'r_DL', 'r_AR', 'r_DH', 'r_AV', 'r_LL', 'r_RR', 'r_LH', 'r_RV', 'r_HH', 'r_VV']
    elif input_method=='prob_15_r':
        inputs = ['r_DD', 'r_AA', 'r_DL', 'r_AR', 'r_DH', 'r_AV', 'r_LL', 'r_RR', 'r_LH', 'r_RV', 'r_HH', 'r_VV', 'r_DR', 'r_DV', 'r_LV']
    
    if pop_method=='raw':
        try:
            inputs += ['d_HandV', 'd_DandA', 'd_RandL']
        except:
            inputs = ['d_HandV', 'd_DandA', 'd_RandL']
    elif pop_method=='diff':
        try:
            inputs += ['HV_DA', 'HV_RL', 'DA_RL']
        except:
            inputs = ['HV_DA', 'HV_RL', 'DA_RL']
    elif pop_method=='rd':
        try:
            inputs += ['d_HandV', 'd_DandA', 'd_RandL', 'HV_DA', 'HV_RL', 'DA_RL']
        except:
            inputs = ['d_HandV', 'd_DandA', 'd_RandL', 'HV_DA', 'HV_RL', 'DA_RL']

    print('inputs', inputs)

    if task=='w':
        df_full = df.copy()
        try: 
            if w_cond:
                df = df.loc[(df['W_min']>=0) & ((df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t3']<0)) & (df['concurrence']> conc_threshold)]
            else:
                df = df.loc[(df['concurrence']> conc_threshold)] # get only entangled states
        except KeyError: # no W value; keep whole df (for testing 102 states from last sem)
            pass
        outputs=['Wp_t1', 'Wp_t2', 'Wp_t3']
        def simplify_targ(df):
            print('number satisfy', len(df))
            print('percent satisfy', len(df)/len(df_full))
            ''' If W' val is negative, set to 1'''
            output_df= df.applymap(lambda x: 1 if x < 0 else 0)
            return output_df

    elif task=='e':
        outputs=['concurrence']
        # randomly remove some entangled states to balance the dataset 2/3 entangled, 1/3 separable
        # alpha = 2*len(df.loc[np.isclose(df['concurrence'], 0, rtol=1e-9)]) / len(df.loc[df['concurrence']>0])
        # print('alpha',alpha)
        # i_to_rem = np.random.choice(df[df['concurrence']>0].index, size=int(len(df[df['concurrence']>0])*(1-alpha)), replace=False)
        # df = df.drop(i_to_rem)
            
        def simplify_targ(df):
            ''' Use concurrence to decide between entangled and separable states '''
            # randomly remove some entangled states to balance the dataset
            output_df_0 = df.applymap(lambda x: 1 if x > 0 else 0) # for the elem at index 0, i.e. is entangled?
            output_df_1 = df.applymap(lambda x: 1 if np.isclose(x,0, rtol=1e-9) else 0) # for the elem at index 1, i.e. is separable?
            print('number entangled', output_df_0.sum())
            print('percent entangled', output_df_0.sum()/len(df))
            output_df = pd.DataFrame()
            output_df['entangled'] = output_df_0
            output_df['separable'] = output_df_1
            return output_df

    def split_data():
        ''' Function to split data into train and test sets.'''
        split_index = int(p*len(df))
        df_train = df.iloc[:split_index, :]
        df_test = df.iloc[split_index:, :]

        X_train = df_train[inputs]
        print('X_train', X_train.head())
        Y_train = simplify_targ(df_train[outputs])
        print('Y_train', Y_train.head())

        X_test = df_test[inputs]
        Y_test = simplify_targ(df_test[outputs])

        X_train, Y_train, X_test, Y_test = X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()

        if normalize:
            X_train /= np.linalg.norm(X_train, axis=0)
            X_test /= X_test/np.linalg.norm(X_test, axis=0)

        # if savename != 'None':
        #     np.save(join(datapath, savename+'_X_train.npy'), X_train.to_numpy())
        #     np.save(join(datapath, savename+'_Y_train.npy'), Y_train.to_numpy())
        #     np.save(join(datapath, savename+'_X_test.npy'), X_test.to_numpy())
        #     np.save(join(datapath, savename+'_Y_test.npy'), Y_test.to_numpy())    
        return X_train, Y_train, X_test, Y_test

    if split: return split_data()
    else: 
        print(df[inputs].to_numpy())
        print(simplify_targ(df[outputs]).to_numpy())
        inputs =  df[inputs].to_numpy()
        outputs = simplify_targ(df[outputs]).to_numpy()
        if normalize:
            inputs = inputs/np.linalg.norm(inputs, axis=0)
            outputs = outputs/np.linalg.norm(outputs, axis=0)
        return inputs, outputs
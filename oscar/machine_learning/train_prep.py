# file to prepare data for ml
import os
from os.path import join
import pandas as pd
import numpy as np

def prepare_data(datapath, savename, method, p=0.8):
    ''' Function to prepare data for training.
    params:
        datapath: path to csv
        savename: what to name the prepped data
        method: 'witness_s' (simplex) or 'witness_j' or witness_js for combined (jones) or 'entangled' for prediction method
        p: fraction of data to use for training
    '''

    # load multiple csvs
    if os.isdir(datapath):
        df= pd.DataFrame()
        for f in os.listdir(datapath):
            if method=='witness_s':
                if f.endswith('.csv') and f.split('_')[0]=='simplex':
                    df = pd.concat([df, pd.read_csv(join(datapath, f))])
            elif method=='witness_j':
                if f.endswith('.csv') and f.split('_')[0]=='jones':
                    df = pd.concat([df, pd.read_csv(join(datapath, f))])
            elif method=='witness_js':
                if f.endswith('.csv'):
                    df = pd.concat([df, pd.read_csv(join(datapath, f))])
    else:
        df= pd.read_csv(datapath)


    other = [] # for other columns to save
    if method=='witness_s' or method=='witness_j' or method=='witness_js':
        inputs = ['HH','HV', 'VH','VV', 'DD', 'DA','AD','AA', 'RR', 'RL','LR','LL']
        outputs=['Wp_t1', 'Wp_t2', 'Wp_t3']
        if method=='witness_s':
            other = ['a','b','c','d','beta', 'gamma', 'delta']
        elif method=='witness_j':
            other = ['theta1', 'theta2', 'alpha1', 'alpha2', 'phi']
        else: # if combined, include all
            other = ['a','b','c','d','beta', 'gamma', 'delta', 'theta1', 'theta2', 'alpha1', 'alpha2', 'phi']
        
        # reassign df so it only has rows corresponding to positive W and >=1 W'<0
        df[(df['W_min']>=0) & ((df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t1']<0))]

    elif method=='entangled':
        # add code here
        outputs = ['min_eig']

    if method=='witness_s' or method=='witness_j':
        def simplify_targ(df):
            ''' If W' val is negative, set to 1'''
            output_df= df.applymap(lambda x: 1 if x < 0 else 0)
            return output_df

    elif method=='entangled':
        output_df_0 = df[outputs].applymap(lambda x: 1 if x < 0 else 0) # for the elem at index 0, i.e. is entangled?
        output_df_1 = df[outputs].applymap(lambda x: 1 if x > 0 else 0) # for the elem at index 1, i.e. is separable?
        output_df = pd.DataFrame()
        output_df['class 0'] = output_df_0
        output_df['class 1'] = output_df_1
        return output_df

    def split_data():
        ''' Function to split data into train and test sets.'''
        split_index = int(p*len(df))
        df_train = df.iloc[:split_index, :]
        df_test = df.iloc[split_index:, :]

        X_train = df_train[inputs]
        Y_train = simplify_targ(df_train[outputs])

        X_test = df_test[inputs]
        Y_test = simplify_targ(df_test[outputs])

        if len(other)==0:
            np.save(join(savename+'_X_train.npy'), X_train.to_numpy())
            np.save(join(savename+'_Y_train.npy'), Y_train.to_numpy())
            np.save(join(savename+'_X_test.npy'), X_test.to_numpy())
            np.save(join(savename+'_Y_test.npy'), Y_test.to_numpy())
            return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()
        else:
            Other_train = df_train[other]
            Other_test = df_test[other]

            np.save(join(savename+'_X_train.npy'), X_train.to_numpy())
            np.save(join(savename+'_Y_train.npy'), Y_train.to_numpy())
            np.save(join(savename+'_X_test.npy'), X_test.to_numpy())
            np.save(join(savename+'_Y_test.npy'), Y_test.to_numpy())
            np.save(join(savename+'_Other_train.npy'), Other_train.to_numpy())
            np.save(join(savename+'_Other_test.npy'), Other_test.to_numpy())
            return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy(), Other_train.to_numpy(), Other_test.to_numpy()
    
    return split_data()
# file to prepare data for ml

from os.path import join
import pandas as pd

def prepare_data(datapath, method='witness_s', p=0.8):
    ''' Function to prepare data for training.
    params:
        datapath: path to csv
        method: 'witness_s' (simplex) or 'witness_j' (jones) or 'entangled' for prediction method
        p: fraction of data to use for training
    '''
    df= pd.read_csv(datapath)

    if method=='witness_s' or method=='witness_j':
        inputs = ['HH','HV', 'VH','VV', 'DD', 'DA','AD','AA', 'RR', 'RL','LR','LL']
        outputs=['Wp_t1', 'Wp_t2', 'Wp_t3']
        if method=='witness_s':
            other = ['a','b','c','d','beta', 'gamma', 'delta']
        else:
            other = ['theta1', 'theta2', 'alpha1', 'alpha2', 'phi']
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

        return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()
    
    return split_data()
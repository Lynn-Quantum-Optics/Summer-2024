# file to revamp becca and laney's neural net from last semester
from os.path import join
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam

# read in and prepare data
DATA_PATH = 'S22_data'

def prepare_data(df_path):
    df1 = pd.read_csv(join(DATA_PATH, df_path))
    # define inputs and outputs for model
    inputs = ['HH probability', 'HV probability', 'VH probability', 'VV probability',
        'DD probability', 'DA probability', 'AD probability', 'AA probability',
        'RR probability', 'RL probability', 'LR probability', 'LL probability']
    outputs = ['XY min', 'YZ min','XZ min']

    # function to simplify the outputs by replacing the most negative value with 1 and setting others to 0
    def simplify_targ(df, outputs):
        df_t = df.T # transpose original to get argmin on each column
        df_out = pd.DataFrame(columns=[i for i in range(len(df_t))])

        def get_out_vec(values):
            targ = values.argmin()
            out_vec = np.zeros(len(outputs))
            out_vec[targ]=1
            return out_vec

        for _, values in df_t.iteritems():
            df_out.loc[len(df_out)] = get_out_vec(values)

        # rename
        # df_out_n = df_out.set_axis(outputs, axis=1)
        return df_out

    # function to build the dataset: p is proportion in training vs test; inputs is list of strings of input states, same for outputs
    def prep_data(df, p, inputs, outputs):
        split_index = int(p*len(df))
        df_train = df.iloc[:split_index, :]
        df_test = df.iloc[split_index:, :]

        X_train = df_train[inputs]
        Y_train = simplify_targ(df_train[outputs], outputs)

        X_test = df_test[inputs]
        Y_test = simplify_targ(df_test[outputs], outputs)

        return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()

    # functions to build and train the network
    X_train, Y_train, X_test, Y_test = prep_data(df1, 0.8, inputs, outputs)

    # save!
    np.save(join(DATA_PATH, 'x_train_'+df_path+'.npy'), X_train)
    np.save(join(DATA_PATH,'y_train_'+df_path+'.npy'), Y_train)
    np.save(join(DATA_PATH,'x_test_'+df_path+'.npy'), X_test)
    np.save(join(DATA_PATH,'y_test_'+df_path+'.npy'), Y_test)

# read in data
def read_data(df_path):
    X_train = np.load(join(DATA_PATH, 'x_train_'+df_path+'.npy'))
    Y_train = np.load(join(DATA_PATH, 'y_train_'+df_path+'.npy'))
    X_test = np.load(join(DATA_PATH, 'x_test_'+df_path+'.npy'))
    Y_test = np.load(join(DATA_PATH, 'y_test_'+df_path+'.npy'))

    return X_train, Y_train, X_test, Y_test

## call to load and save dfs ##
# prepare_data('all_qual_20000_1.csv')
X_train1, Y_train1, X_test1, Y_test1 = read_data('all_qual_20000_1.csv')

## model creation ##
output_len = 3
def build_model_test(size=128, dropout=0, learning_rate=0.01):
    model = Sequential()

    model.add(layers.Dense(size))
    model.add(layers.Dense(size))
    model.add(layers.Dense(size))
    model.add(layers.Dense(size))

    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_len))

    # return len of class size
    model.add(layers.Dense(output_len))
    model.add(layers.Activation('softmax'))

    optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

def train_manual(X_train, Y_train, X_test, Y_test):
    global model
    model = build_model_test()
    
    #now run training
    history = model.fit(
    X_train, Y_train,
    batch_size = 100,
    validation_data=(X_test, Y_test),
    epochs=100
    )

## call training ##
train_manual(X_train1, Y_train1, X_test1, Y_test1)
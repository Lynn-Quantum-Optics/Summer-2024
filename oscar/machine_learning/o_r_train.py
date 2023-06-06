# file to convert the dfs into readable input and output arrays
import numpy as np
import pandas as pd
from os.path import join
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import keras
from tqdm import tqdm

def prepare_data(df, p):
    '''
    Takes as input a full dataframe generated with o_r_datagen.py and returns X_train/test, Y_train/test matrices, split according to the fraction dedicated to training p
    '''
    df = df.sample(frac=1, random_state=47) # shuffle the rows
    input_columns = list(df.columns[1:-2]) # don't want the min eigenval or purity
    input_df = df[input_columns]

    # output_df = np.where(df['min_eig']>0, 1, 0) # if true (i.e. separable), then puts 1; else, entangled
    # make 1 hot encoded vector
    output_df_0 = df[['min_eig']].applymap(lambda x: 1 if x < 0 else 0) # for the elem at index 0, i.e. is entangled?
    output_df_1 = df[['min_eig']].applymap(lambda x: 1 if x > 0 else 0) # for the elem at index 1, i.e. is separable?
    output_df = pd.DataFrame()
    output_df['entangled'] = output_df_0
    output_df['separable'] = output_df_1

    print('num entangled', np.sum(output_df) / len(output_df))

    split_index = int(p*len(df))
    X_train = input_df.iloc[:split_index, :].to_numpy()
    X_test = input_df.iloc[split_index:, :].to_numpy()
    Y_train = output_df[:split_index].to_numpy()
    Y_test = output_df[split_index:].to_numpy()

    return X_train, Y_train, X_test, Y_test

def do_xgboost(X_train, Y_train, X_test, Y_test, M=1000, lr=0.3):
    '''
    Run the XGBoost algorithm given M trees and lr learning rate; returns model
    '''
    model = XGBClassifier(n_estimators=M, eta=lr, tree_method='hist') # use tree_method = 'gpu_hist' if running on GPU
    # train the model: early_stopping_rounds will stop training if val loss does not improve after that many rounds
    model.fit(X_train, Y_train, early_stopping_rounds = 10, eval_set = [(X_test, Y_test)], verbose=True) 

    return model

def do_nn(X_train, Y_train, X_test, Y_test, epochs):
    model = keras.Sequential([
    keras.layers.Dense(36, activation="relu"),
    keras.layers.Dense(72, activation="relu"),
    keras.layers.Dense(36, activation="relu"),
    keras.layers.Dense(6, activation="relu"),
    keras.layers.Dense(2, activation="softmax")])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
    return model

def evaluate_perf(epsilon, model, X_train, Y_train, X_test, Y_test):
    '''
    Computes accuracy of model on both train and test data as well as confusion matrix; epsilon is the threshold between entangled (0) and separable (1)
    '''
    Y_pred_test = model.predict(X_test)
    Y_pred_train = model.predict(X_train)

    if len(Y_pred_test[0]) ==1:
        Y_pred_test = np.where(Y_pred_test > epsilon, 1, 0)
        Y_pred_train = np.where(Y_pred_train > epsilon, 1, 0)

    else:
        Y_pred_test = np.where(Y_pred_test[:, 1:] > epsilon, 1, 0)
        Y_pred_train = np.where(Y_pred_train[:, 1:] > epsilon, 1, 0)

    test_acc = np.sum(np.where(Y_pred_test==Y_test, 1, 0)) / len(Y_test)
    train_acc = np.sum(np.where(Y_pred_train==Y_train, 1, 0)) / len(Y_train)

    return test_acc, train_acc

    # get confusion matrix


def plot_epsilon(xgb_model, nn_model, figname):
    epsilon_ls = np.linspace(.1, 0.9, 5)
    test_acc_xgb = []
    train_acc_xgb = []
    test_acc_nn = []
    train_acc_nn = []
    for _, epsilon in enumerate(tqdm(epsilon_ls)):
        xgb_acc = evaluate_perf(epsilon, xgb_model, X_train, Y_train, X_test, Y_test)
        nn_acc = evaluate_perf(epsilon, nn_model, X_train, Y_train, X_test, Y_test)

        test_acc_xgb.append(xgb_acc[0])
        train_acc_xgb.append(xgb_acc[1])
        test_acc_nn.append(nn_acc[0])
        train_acc_nn.append(nn_acc[1])
    
    plt.figure(figsize=(10,7))
    plt.plot(epsilon_ls, test_acc_nn, label='Test NN')
    plt.plot(epsilon_ls, train_acc_nn, label='Train NN')
    plt.plot(epsilon_ls, test_acc_xgb, label='Test XGB')
    plt.plot(epsilon_ls, train_acc_xgb, label='Train XGB')
    plt.xlabel('$\epsilon$', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Comparison of XGB and NN', fontsize=16)
    plt.legend()
    plt.savefig(join(DATA_PATH, figname+'.pdf'))
    plt.show()


## load data ##
DATA_PATH='RO_data'

df_12 = pd.read_csv(join('RO_data', 'df_12.csv'))
X_train, Y_train, X_test, Y_test = prepare_data(df_12, 0.9)
xgb12 = do_xgboost(X_train, Y_train, X_test, Y_test, M=2000, lr=0.01)
nn12 = do_nn(X_train, Y_train, X_test, Y_test, epochs=20)
plot_epsilon(xgb12, nn12, 'first')






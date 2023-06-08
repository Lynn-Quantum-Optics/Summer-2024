# file to perform hyperparameter sweep for XGBOOST and NN

import wandb
import numpy as np
from xgboost import XGBClassifier

## compute accuracy ##
def evaluate_perf(model, method, X_train, Y_train, X_test, Y_test):
    ''' Function to measure accuracy on both train and test data.
    Params:
        model: trained model
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate on
        '''
    Y_pred_test = model.predict(X_test)
    Y_pred_train = model.predict(X_train)

    N_correct_test = 0
    N_correct_train = 0
    for i, y_pred in enumerate(Y_pred_test):
        if Y_test[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
            N_correct_test+=1
    
    for i, y_pred in enumerate(Y_pred_train):
        if Y_train[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
            N_correct_train+=1

    if method == 'witness':
        Ud = Y_test.sum(axis=1) # undetectables: count the number of states w negative witness value
        return [N_correct_test / (len(Y_pred_test) - len(Ud[Ud==0])), N_correct_train / (len(Y_pred_train) - len(Ud[Ud==0]))]
    elif method == 'entangled':
        return [N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)]
        


#######################################################
## XGBOOST ##
''' Total list of params: https://xgboost.readthedocs.io/en/stable/parameter.html'''

xgb_sweep_config = {
    "method": "bayes",
    "metric": {"name": "mse", "goal": "minimize"},
    "parameters": {
        "max_depth": {"values": [1, 2, 5, 10]},
        "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 0.9},
        "n_estimators": {"distribution": "uniform", "min": 1000, "max": 5000},
        "early_stopping": {"distribution": "uniform", "min": 5, "max": 30}
        },
    }

def train_xgb(method, X_train, Y_train, X_test, Y_test):
    ''' Function to run wandb sweep for XGBOOST. 
    Adapted from https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py 
    params:
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate on
    '''
    wandb.init() # initialize wandb client

    # define the model
    model = XGBClassifier(
        max_depth=wandb.config.max_depth,
        learning_rate=wandb.config.learning_rate,
        n_estimators=wandb.config.n_estimators,
    )

    # fit the model
    model.fit(X_train, Y_train, tree_method='hist', early_stopping_rounds = wandb.config.early_stopping, random_state=42, eval_set=[(X_test, Y_test)], callbacks=[wandb.xgboost.WandbCallback()])

    # log test accuracy to wandb
    acc = evaluate_perf(model, method, X_train, Y_train, X_test, Y_test)[0]
    wandb.log({"acc": acc})

def custom_train_xgb(method, X_train, Y_train, X_test, Y_test, max_depth=6, learning_rate=0.3, n_estimators=1000, early_stopping=10):
    ''' Function to run XGBOOST with custom hyperparameters.
    params:
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate on
    '''
    # define the model
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators= n_estimators,
    )

    # fit the model
    model.fit(X_train, Y_train, tree_method='hist', early_stopping_rounds = early_stopping, random_state=42, eval_set=[(X_test, Y_test)])

    # print results
    acc = evaluate_perf(model, method, X_train, Y_train, X_test, Y_test)
    print('Accuracy on test, train', acc)

#######################################################
## NN ##
def train_nn(X_train, Y_train, X_test, Y_test):
    # code here
   pass


## run on our data ##
if __name__=='__main__':
    import pandas as pd
    # load data here

    def run_sweep(do_xgb=True):
        if do_xgb:
            sweep_id = wandb.sweep(xgb_sweep_config, project="Lynn Quantum Optics")
            wandb.agent(sweep_id=sweep_id, function=train_xgb)
        else:
            sweep_id = wandb.sweep(nn_sweep_config, project="Lynn Quantum Optics")
            wandb.agent(sweep_id=sweep_id, function=train_nn)
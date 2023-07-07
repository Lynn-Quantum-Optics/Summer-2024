# file to perform hyperparameter sweep for XGBOOST and NN

import wandb
import numpy as np
from xgboost import XGBClassifier
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam

#######################################################
## XGBOOST ##
''' Total list of params: https://xgboost.readthedocs.io/en/stable/parameter.html'''
def train_xgb():
    ''' Function to run wandb sweep for XGBOOST. 
    Adapted from https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py 
    params:
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate on
    '''
    wandb.init(config=xgb_sweep_config) # initialize wandb client

    # define the model
    model = XGBClassifier(
        max_depth=wandb.config.max_depth,
        learning_rate=wandb.config.learning_rate,
        n_estimators=wandb.config.n_estimators,
    )

    # fit the model
    model.fit(X_train, Y_train, early_stopping_rounds = wandb.config.early_stopping, eval_set=[(X_test, Y_test)], callbacks=[wandb.xgboost.WandbCallback(log_model=True)])
    # early_stopping_rounds = int(wandb.config.early_stopping)

    # log test accuracy to wandb
    # val_acc = evaluate_perf(model, X_train, Y_train, X_test, Y_test)[0]
    # wandb.log({"val_acc": val_acc})

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
    # acc = evaluate_perf(model, method, X_train, Y_train, X_test, Y_test)
    # print('Accuracy on test, train', acc)

#######################################################
## NN ##
def train_nn1h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn1h_sweep_config) # initialize wandb client

    
    def build_model(size, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('softmax'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        return model
    
    model = build_model(wandb.config.size,wandb.config.learning_rate)
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = wandb.config.batch_size,
        validation_data=(X_test,Y_test),
        epochs=50,
        callbacks=[wandb.keras.WandbCallback()] #use callbacks to have w&b log stats; will automatically save best model                     
      )

def train_nn3h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn3h_sweep_config) # initialize wandb client

    
    def build_model(size1, size2, size3, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1))
        model.add(layers.Dense(size2))
        model.add(layers.Dense(size3))

        # model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('softmax'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        return model
    
    model = build_model(wandb.config.size_1,  wandb.config.size_2, wandb.config.size_3, wandb.config.learning_rate)
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = wandb.config.batch_size,
        validation_data=(X_test,Y_test),
        epochs=50,
        callbacks=[wandb.keras.WandbCallback()] #use callbacks to have w&b log stats; will automatically save best model                     
      )

def train_nn5h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn5h_sweep_config) # initialize wandb client

    
    def build_model(size1, size2, size3, size4, size5, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1))
        model.add(layers.Dense(size2))
        model.add(layers.Dense(size3))
        model.add(layers.Dense(size4))
        model.add(layers.Dense(size5))

        # model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('softmax'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        return model
    
    model = build_model(wandb.config.size_1,  wandb.config.size_2, wandb.config.size_3, 
              wandb.config.size_4, wandb.config.size_5, wandb.config.learning_rate)
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = wandb.config.batch_size,
        validation_data=(X_test,Y_test),
        epochs=50,
        callbacks=[wandb.keras.WandbCallback()] #use callbacks to have w&b log stats; will automatically save best model                     
      )


## run on our data ##
if __name__=='__main__':
    from train_prep import prepare_data

    file = input('Enter file name: ')
    input_method = input('Enter input method: ')
    task = input('Enter task: ')
    identifier = input('Enter identifier ([randomtype][method]_[attempt]): ')
    savename= identifier+'_'+task+'_'+input_method

    # load data here
    DATA_PATH = 'random_gen/data'
    X_train, Y_train, X_test, Y_test = prepare_data(datapath=DATA_PATH, file=file, input_method=input_method, task=task)

    ## sweep configs ##
    if task=='w':
        xgb_sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "max_depth": {"distribution": "int_uniform", "min":  1, "max": 20},
            "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 0.5},
            "n_estimators": {"distribution": "int_uniform", "min":  500, "max": 10000},
            "early_stopping": {"distribution": "int_uniform", "min": 5, "max": 40}
        },
        }

        nn1h_sweep_config = {
        'method': 'random',
        'name': 'val_accuracy',
        'goal': 'maximize',
        'parameters':{
        # for build_dataset
        'batch_size': {
        'values': [x for x in range(256, 4481, 32)]
        },
        'size': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 1800
        },
        'learning_rate':{
            #uniform distribution between 0 and 1
            'distribution': 'uniform', 
            'min': 1e-5,
            'max': 0.5
        }
        },
        }

        nn3h_sweep_config = {
        'method': 'random',
        'name': 'val_accuracy',
        'goal': 'maximize',
        'parameters':{
        # for build_dataset
        'batch_size': {
        'values': [x for x in range(256, 4481, 32)]
        },
        'size_1': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 1800
        },
        'size_2': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 1800
        },'size_3': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 1800,
        },
        # 'dropout': {
        # 'distribution': 'uniform',
        # 'min': 0,
        # 'max': 0.6
        # },
        'learning_rate':{
            #uniform distribution between 0 and 1
            'distribution': 'uniform', 
            'min': 1e-5,
            'max': 0.5
        }
        },
        }

        nn5h_sweep_config = {
            'method': 'random',
            'name': 'val_accuracy',
            'goal': 'maximize',
            'metric':'val_accuracy',
        'parameters':{
            # for build_dataset
            'batch_size': {
            'values': [x for x in range(256, 4481, 32)]
            },
            'size_1': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },
            'size_2': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },'size_3': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },'size_4': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },'size_5': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },
            # 'dropout': {
            # 'distribution': 'uniform',
            # 'min': 0,
            # 'max': 0.6
            # },
            'learning_rate':{
                #uniform distribution between 0 and 1
                'distribution': 'uniform', 
                'min': 1e-5,
                'max': 0.5
            }
        },
        }

    elif task=='e':
        xgb_sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "max_depth": {"distribution": "int_uniform", "min":  3, "max": 10},
            "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 0.1},
            "n_estimators": {"distribution": "int_uniform", "min":  500, "max": 8500},
            "early_stopping": {"distribution": "int_uniform", "min": 5, "max": 30}
        },
        }

        nn1h_sweep_config = {
        'method': 'random',
        'name': 'val_accuracy',
        'goal': 'maximize',
        'parameters':{
        'epochs': {
        'distribution': 'int_uniform',
        'min': 20,
        'max': 100
        },
        # for build_dataset
        'batch_size': {
        'values': [x for x in range(256, 4481, 32)]
        },
        'size': {
        'value': 50
        },
        'learning_rate':{
            #uniform distribution between 0 and 1
            'distribution': 'uniform', 
            'min': 1e-5,
            'max': 0.7
        }
        },
        }

        nn3h_sweep_config = {
        'method': 'random',
        'name': 'val_accuracy',
        'goal': 'maximize',
        'parameters':{
        'epochs': {
        'distribution': 'int_uniform',
        'min': 20,
        'max': 100
        },
        # for build_dataset
        'batch_size': {
        'values': [x for x in range(256, 4481, 32)]
        },
        'size_1': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 300
        },
        'size_2': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 300
        },'size_3': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 300,
        },
        'dropout': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.6
        },
        'learning_rate':{
            #uniform distribution between 0 and 1
            'distribution': 'uniform', 
            'min': 1e-5,
            'max': 0.7
        }
        },
        }

        nn5h_sweep_config = {
            'method': 'random',
            'name': 'val_accuracy',
            'goal': 'maximize',
            'metric':'val_accuracy',
        'parameters':{
            'epochs': {
            'value': 50
            },
            # for build_dataset
            'batch_size': {
            'values': [x for x in range(256, 4481, 32)]
            },
            'size_1': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },
            'size_2': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },'size_3': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },'size_4': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },'size_5': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 1800
            },
            'dropout': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.6
            },
            'learning_rate':{
                #uniform distribution between 0 and 1
                'distribution': 'uniform', 
                'min': 1e-5,
                'max': 0.5
            }
        },
        }

    

    def run_sweep(wtr=0):
        ''' Function to run hyperparam sweep.
        Params:
            wtr: int, 0 for XGB, 1 for NN5H, 2 for NN3H
        '''
        if wtr==0:
            sweep_id = wandb.sweep(xgb_sweep_config, project='LQO-XGB_'+savename)
            wandb.agent(sweep_id=sweep_id, function=train_xgb)
        elif wtr==1:
            sweep_id = wandb.sweep(nn1h_sweep_config, project='LQO-NN1H_'+savename)
            wandb.agent(sweep_id=sweep_id, function=train_nn1h) 
        elif wtr==3:
            sweep_id = wandb.sweep(nn3h_sweep_config, project='LQO-NN3H_'+savename)
            wandb.agent(sweep_id=sweep_id, function=train_nn3h)
        elif wtr==5:
            sweep_id = wandb.sweep(nn5h_sweep_config, project='LQO-NN5H_'+savename)
            wandb.agent(sweep_id=sweep_id, function=train_nn5h)
        else:
            raise ValueError('wtr must be 0, 1, 3, or 5.')
    wtr = int(input('Enter 0 for XGB, 1 for NN1H, 3 for NN3H:'))
    run_sweep(wtr)
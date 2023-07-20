# file to perform hyperparameter sweep for XGBOOST and NN

import wandb
import numpy as np
from xgboost import XGBRegressor
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

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
    model = XGBRegressor(
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

def custom_train_xgb(max_depth=10, learning_rate=0.3, n_estimators=1000, early_stopping=10):
    ''' Function to run XGBOOST with custom hyperparameters.
    params:
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate ond
        defauls for ref:
            max_depth=6, learning_rate=0.3, n_estimators=1000, early_stopping=10
    '''
    # define the model
    try:
        model = XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators= n_estimators,
            tree_method = 'hist'
        )
    except:
        print('unable to load gpu_hist, using default')
        model = XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators= n_estimators
        )

    # fit the model
    model.fit(X_train, Y_train,early_stopping_rounds = early_stopping, eval_set=[(X_test, Y_test)])

    # print results
    # acc = evaluate_perf(model, method, X_train, Y_train, X_test, Y_test)
    # print('Accuracy on test, train', acc)
    return model

#######################################################
## NN ##
def train_nn1h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn1h_sweep_config) # initialize wandb client

    
    def build_model(size, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size, activation='relu'))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('sigmoid'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

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
    return  model
    
def custom_train_nn1h(size, learning_rate, epochs=50, batch_size=256):
    ''' Function to run wandb sweep for NN.'''
    
    def build_model(size, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size, activation='relu'))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('sigmoid'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model
    
    model = build_model(size,learning_rate)
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = batch_size,
        validation_data=(X_test,Y_test),
        epochs=epochs,                   
      )
    return model

def train_nn3h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn3h_sweep_config) # initialize wandb client

    
    def build_model(size1, size2, size3, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1, activation='relu'))
        model.add(layers.Dense(size2, activation='relu'))
        model.add(layers.Dense(size3, activation='relu'))

        # model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('sigmoid'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

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

def custom_train_nn3h(size1, size2, size3, learning_rate, epochs=50, batch_size=256):
    ''' Function to run wandb sweep for NN.'''
    
    def build_model(size1, size2, size3, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1, activation='relu'))
        model.add(layers.Dense(size2, activation='relu'))
        model.add(layers.Dense(size3, activation='relu'))

        # model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('sigmoid'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model
    
    model = build_model(size1, size2, size3, learning_rate)
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = batch_size,
        validation_data=(X_test,Y_test),
        epochs=epochs                    
      )
    return model


def train_nn5h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn5h_sweep_config) # initialize wandb client

    
    def build_model(size1, size2, size3, size4, size5, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1, activation='relu'))
        model.add(layers.Dense(size2, activation='relu'))
        model.add(layers.Dense(size3, activation='relu'))
        model.add(layers.Dense(size4, activation='relu'))
        model.add(layers.Dense(size5, activation='relu'))

        # model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('sigmoid'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

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

def custom_train_nn5h(size1, size2, size3, size4, size5, learning_rate, epochs=50, batch_size=256):
    ''' Function to run wandb sweep for NN.'''
    
    def build_model(size1, size2, size3, size4, size5, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1, activation='relu'))
        model.add(layers.Dense(size2, activation='relu'))
        model.add(layers.Dense(size3, activation='relu'))
        model.add(layers.Dense(size4, activation='relu'))
        model.add(layers.Dense(size5, activation='relu'))

        # model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('sigmoid'))

        optimizer = Adam(learning_rate = learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model
    
    model = build_model(size1, size2, size3, size4, size5, learning_rate) 
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = batch_size,
        validation_data=(X_test,Y_test),
        epochs=epochs                   
      )
    return model


## run on our data ##
if __name__=='__main__':
    from os.path import join

    from train_prep import prepare_data
    from ml_comp import eval_perf
    op = int(input('0 for custom run/sweep, 1 for wandb, and 2 for retraining model based on mistakes in tv dataset: '))

    if op==1:
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
    elif op==0:
        # file = 'roik_True_4000000_r_os_tv.csv'
        file = 'roik_4m_wpop_rd.csv'
        task = input('w or e for task: ')
        if task=='e':
            num = int(input('Enter number of prob inputs:'))
            input_method = 'prob_%i'%num
        else:
            input_method = input('Enter input method: ')
            pop_method = input('Enter pop method: ')
        do_sweep = bool(int(input('Enter 1 to do sweep, 0 to train single instance:')))
        trial = int(input('Enter trial number:'))
        identifier = 'r4_s0_%i'%trial
        savename= identifier+'_'+task+'_'+input_method
        # load data here
        DATA_PATH = 'random_gen/data'
        X_train, Y_train, X_test, Y_test = prepare_data(datapath=DATA_PATH, file=file, input_method=input_method, pop_method = pop_method, task=task)

        if do_sweep:   
            wtr = int(input('Which model to run? 0 for XGB, 1 for NN1H, 3 for NN3H, 5 for NN5H:'))
            if wtr==0:
                lr_ls = np.linspace(0.01, 1, 10)
                n_est_ls = np.linspace(1000, 10000, 10)
                n_est_ls = n_est_ls.astype(int)
                n_est_ls = np.unique(n_est_ls)
                max_depth_ls = np.linspace(5, 20, 5)
                max_depth_ls = max_depth_ls.astype(int)
                max_depth_ls = np.unique(max_depth_ls)
                early_stopping_ls = np.linspace(5, 50, 5)
                early_stopping_ls = early_stopping_ls.astype(int)
                early_stopping_ls = np.unique(early_stopping_ls)
                best_acc = 0
                best_lr = None
                best_n_est = None
                best_max_depth = None
                best_early_stopping = None
                best_model = None
                for lr in lr_ls:
                    for n_est in n_est_ls:
                        for max_depth in max_depth_ls:
                            for early_stopping in early_stopping_ls:
                                try:
                                    xgb = custom_train_xgb(n_estimators=int(n_est), learning_rate=lr, max_depth=int(max_depth), early_stopping=int(early_stopping))
                                    df= eval_perf(xgb, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)])
                                    print('n_est', n_est, 'lr', lr, 'max_depth', max_depth, 'early_stopping', early_stopping)
                                    print('acc', df['acc'].values[0], '\nbest_acc', best_acc)
                                    if df['acc'].values[0] > best_acc:
                                        best_acc = df['acc'].values[0]
                                        best_lr = lr
                                        best_n_est = n_est
                                        best_max_depth = max_depth
                                        best_early_stopping = early_stopping
                                        best_model = xgb
                                        print('Best acc: ', best_acc)
                                        print('Best lr: ', best_lr)
                                        print('Best n_est: ', best_n_est)
                                        print('Best max_depth: ', best_max_depth)
                                        print('Best early_stopping: ', best_early_stopping)

                                except KeyboardInterrupt:
                                    print('Best acc: ', best_acc)
                                    print('Best lr: ', best_lr)
                                    print('Best n_est: ', best_n_est)
                                    print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                                    best_model.save_model(join('random_gen', 'models', savename+'_'+f'xgb_{n_est}_{lr}_{max_depth}_{early_stopping}'+'.json'))
                                    
                print('Best acc: ', best_acc)
                print('Best lr: ', best_lr)
                print('Best n_est: ', best_n_est)
                print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                best_model.save_model(join('random_gen', 'models', savename+'_'+f'xgb_{n_est}_{lr}_{max_depth}_{early_stopping}'+'.json'))
            elif wtr==1:   
                lr_ls = np.linspace(0.0001, 1, 10)
                size_ls = np.linspace(20, 200, 10)
                size_ls = size_ls.astype(int)
                size_ls = np.unique(size_ls)
                epochs = 100
                best_acc = 0
                best_lr = None
                best_size = None
                best_model = None
                for lr in lr_ls:
                    for size in size_ls:
                        try:
                            nn1 = custom_train_nn1h(size=int(size), learning_rate=lr, epochs=epochs)
                            df= eval_perf(nn1, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)])
                            print('size', size, 'lr', lr, 'epochs', epochs)
                            print('acc', df['acc'].values[0], '\nbest_acc', best_acc)
                            if df['acc'].values[0] > best_acc:
                                best_acc = df['acc'].values[0]
                                best_lr = lr
                                best_size = size
                                best_model = nn1
                                print('Best acc: ', best_acc)
                                print('Best lr: ', best_lr)
                                print('Best size: ', best_size)
                        except KeyboardInterrupt:
                            print('Best lr: ', best_lr)
                            print('Best size: ', best_size)
                            print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                            best_model.save(join('random_gen', 'models', savename+'_'+f'nn1_{size}_{lr}_{epochs}'+'.h5'))
                print('Best acc: ', best_acc)
                print('Best lr: ', best_lr)
                print('Best size: ', best_size)
                print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                best_model.save(join('random_gen', 'models', savename+'_'+f'nn1_{size}_{lr}_{epochs}'+'.h5'))
            elif wtr==3:
                lr_ls = np.linspace(0.0001, 1, 10)
                size_ls = np.linspace(20, 300, 5)
                size_ls = size_ls.astype(int)
                size_ls = np.unique(size_ls)
                epochs=100
                best_acc = 0
                best_lr = None
                best_size1 = None
                best_size2 = None
                best_size3 = None
                for lr in lr_ls:
                    for size1 in size_ls:
                        for size2 in size_ls:
                            for size3 in size_ls:
                                try:
                                    nn3 = custom_train_nn3h(size1=int(size1), size2=int(size2), size3=int(size3), learning_rate=lr, epochs=epochs)
                                    df= eval_perf(nn3, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)])
                                    print('size1', size1, 'size2', size2, 'size3', size3, 'lr', lr, 'epochs', epochs)
                                    print('acc', df['acc'].values[0], '\nbest_acc', best_acc)
                                    if df['acc'].values[0] > best_acc:
                                        best_acc = df['acc'].values[0]
                                        best_lr = lr
                                        best_size1 = size1
                                        best_size2 = size2
                                        best_size3 = size3
                                        best_model = nn3
                                        print('Best acc: ', best_acc)
                                        print('Best lr: ', best_lr)
                                        print('Best size1: ', best_size1)
                                        print('Best size2: ', best_size2)
                                        print('Best size3: ', best_size3)
                                except KeyboardInterrupt:
                                    print('Best lr: ', best_lr)
                                    print('Best size1: ', best_size1)
                                    print('Best size2: ', best_size2)
                                    print('Best size3: ', best_size3)
                                    print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                                    best_model.save(join('random_gen', 'models', savename+'_'+f'nn3_{size1}_{size2}_{size3}_{lr}_{epochs}'+'.h5'))
                print('Best acc: ', best_acc)
                print('Best lr: ', best_lr)
                print('Best size1: ', best_size1)
                print('Best size2: ', best_size2)
                print('Best size3: ', best_size3)
                print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                best_model.save(join('random_gen', 'models', savename+'_'+f'nn3_{size1}_{size2}_{size3}_{lr}_{epochs}'+'.h5'))
            elif wtr==5:
                lr_ls = np.linspace(0.0001, 1, 10)
                size_ls = np.linspace(20, 200, 5)
                size_ls = size_ls.astype(int)
                size_ls = np.unique(size_ls)
                epochs=100
                best_acc = 0
                best_lr = None
                best_size1 = None
                best_size2 = None
                best_size3 = None
                best_size4 = None
                best_size5 = None
                best_model = None
                for lr in lr_ls:
                    for size1 in size_ls:
                        for size2 in size_ls:
                            for size3 in size_ls:
                                for size4 in size_ls:
                                    for size5 in size_ls:
                                        try:
                                            nn5 = custom_train_nn5h(size1=int(size1), size2=int(size2), size3=int(size3), size4=int(size4), size5=int(size5), learning_rate=lr, epochs=epochs)
                                            df= eval_perf(nn5, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)])
                                            print('size1', size1, 'size2', size2, 'size3', size3, 'size4', size4, 'size5', size5, 'lr', lr, 'epochs', epochs)
                                            print('acc', df['acc'].values[0], '\nbest_acc', best_acc)
                                            if df['acc'].values[0] > best_acc:
                                                best_acc = df['acc'].values[0]
                                                best_lr = lr
                                                best_size1 = size1
                                                best_size2 = size2
                                                best_size3 = size3
                                                best_size4 = size4
                                                best_size5 = size5
                                                best_model = nn5
                                                print('Best acc: ', best_acc)
                                                print('Best lr: ', best_lr)
                                                print('Best size1: ', best_size1)
                                                print('Best size2: ', best_size2)
                                                print('Best size3: ', best_size3)
                                                print('Best size4: ', best_size4)
                                                print('Best size5: ', best_size5)
                                                print('Best acc: ', best_acc)
                                        except:
                                            print('Best lr: ', best_lr)
                                            print('Best size1: ', best_size1)
                                            print('Best size2: ', best_size2)
                                            print('Best size3: ', best_size3)
                                            print('Best size4: ', best_size4)
                                            print('Best size5: ', best_size5)
                                            print('Best acc: ', best_acc)
                                            print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                                            best_model.save(join('random_gen', 'models', savename+'_'+f'nn5_{size1}_{size2}_{size3}_{size4}_{size5}_{lr}_{epochs}'+'.h5'))
                print('Best lr: ', best_lr)
                print('Best size1: ', best_size1)
                print('Best size2: ', best_size2)
                print('Best size3: ', best_size3)
                print('Best size4: ', best_size4)
                print('Best size5: ', best_size5)
                print('performance on 400k unknown', eval_perf(best_model, identifier+'_'+str(wtr), file_ls = ['roik_True_400000_r_os_t.csv'], task=task, input_method=input_method)['acc'].values[0])
                best_model.save(join('random_gen', 'models', savename+'_'+f'nn5_{size1}_{size2}_{size3}_{size4}_{size5}_{lr}_{epochs}'+'.h5'))
                print('Best lr: ', best_lr)

        else:
            wtr = int(input('Enter 0 for XGB, 1 for NN1H, 3 for NN3H, or 5 for NN5H:'))
            # file_ls = ['roik_True_400000_r_os_t.csv']
            file_ls  = ['roik_400k_wpop_rd.csv', 'roik_400k_extra_wpop_rd.csv'] # this is for testing
            if wtr==0:
                n_estimators = int(input('Enter n_estimators:'))
                learning_rate = float(input('Enter learning_rate:'))
                max_depth = int(input('Enter max_depth:'))
                early_stopping = int(input('Enter early_stopping:'))
                xgb = custom_train_xgb(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, early_stopping=early_stopping)
                print('val', eval_perf(xgb, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)]))
                print(eval_perf(xgb, identifier+'_'+str(wtr), file_ls = file_ls, task=task, input_method=input_method))
                xgb.save_model(join('random_gen', 'models', savename+'_'+f'xgb_{n_estimators}_{learning_rate}_{max_depth}_{early_stopping}'+'.json'))
            elif wtr==1:
                size = int(input('Enter size:'))
                learning_rate = float(input('Enter learning_rate:'))
                epochs=100
                nn1 = custom_train_nn1h(size=size, learning_rate = learning_rate, batch_size=256, epochs=epochs)
                print('val', eval_perf(nn1, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)]))
                print(eval_perf(nn1, identifier+'_'+str(wtr), file_ls = file_ls, task=task, input_method=input_method))
                nn1.save(join('random_gen', 'models', savename+f'_{size}_{learning_rate}_{epochs}.h5'))
            elif wtr==3:
                size1 = int(input('Enter size1:'))
                size2 = int(input('Enter size2:'))
                size3 = int(input('Enter size3:'))
                learning_rate = float(input('Enter learning_rate:'))
                epochs=100
                nn3 = custom_train_nn3h(size1=size1, size2=size2, size3=size3, learning_rate = learning_rate, batch_size=256, epochs=epochs)
                print('val', eval_perf(nn3, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)]))
                print(eval_perf(nn3, identifier+'_'+str(wtr), file_ls = file_ls, task=task, input_method=input_method))
                nn3.save(join('random_gen', 'models', savename+f'_{size1}_{size2}_{size3}_{learning_rate}_{epochs}.h5'))
            elif wtr==5:
                size1 = int(input('Enter size1:'))
                size2 = int(input('Enter size2:'))
                size3 = int(input('Enter size3:'))
                size4 = int(input('Enter size4:'))
                size5 = int(input('Enter size5:'))
                learning_rate = float(input('Enter learning_rate:'))
                epochs=100
                nn5 = custom_train_nn5h(size1=size1, size2=size2, size3=size3, size4=size4, size5=size5, learning_rate = learning_rate, batch_size=256, epochs=epochs)
                print('val', eval_perf(nn5, identifier+'_'+str(wtr), data_ls = [(X_test, Y_test)], task=task, input_method=input_method))
                print(eval_perf(nn5, identifier+'_'+str(wtr), file_ls =file_ls, task=task, input_method=input_method))
                nn5.save(join('random_gen', 'models', savename+f'_{size1}_{size2}_{size3}_{size4}_{size5}_{learning_rate}_{epochs}.h5'))
    elif op==2:
        # for now just load in nn5
        MODEL_PATH = 'random_gen/models'
        DATA_PATH = 'random_gen/data'
        from keras.models import load_model
        from ml_comp import get_labels, eval_perf

        nn5 = load_model(join(MODEL_PATH, 'r4_s0_7_w_prob_9_200_200_200_200_200_0.0001_100.h5'))
        # look at performance on training and validation set
        # tv_file = 'roik_True_4000000_tv.csv'
        tv_file = 'roik_True_10000_extra.csv'
        _, _, tv_X, tv_Y = prepare_data(datapath=DATA_PATH, file=tv_file, input_method='prob_9', task='w', split=True)
        tv_predY = nn5.predict(tv_X)
        tv_predY = get_labels(tv_predY, task='w')
        dot = np.einsum('ij,ij->i', tv_Y, tv_predY)
        # get baseline performance
        print('baseline', np.sum(dot)/len(dot))
        # double check eval_perf
        # now look at performance on test set
        file_ls = ['roik_True_400000_s0_extra0.csv', 'roik_True_400000_r_os_t.csv']
        print(eval_perf(nn5, 'nn5', file_ls=file_ls, file_names = ['Test_extra', 'Test'], task='w', input_method='prob_9'))
        # find inputs that resulted into incorrect outputs
        bad_tvX = tv_X[dot==0]
        bad_tvY = tv_Y[dot==0]
        # now resume training
        epochs = 100
        batch_size = 256
        nn5.fit(tv_X, tv_Y, batch_size=batch_size, epochs=epochs)
        # nn5.fit(bad_tvX, bad_tvY, batch_size=batch_size, epochs=epochs)
        # nn5.save(join(MODEL_PATH, 'r4_s0_7_w_prob_9_200_200_200_200_200_0.0001_100_retraintv.h5'))
        print(eval_perf(nn5, 'nn5_retrain', file_ls=file_ls, file_names = ['Test_extra', 'Test'], task='w', input_method='prob_9'))

        


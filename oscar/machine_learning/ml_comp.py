# file to compute performance of models on new data, specifically 400k Roik and 700k Matlab
from os.path import join
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import keras
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train_prep import prepare_data

def get_labels(Y_pred, task, eps=None):
    ''' Function to assign labels based on argmax per row.
    Params:
        Y_pred: array of predictions after calling model.predict(X)
        task: 'w' or 'e'
        eps: threshold for assigning label if task is 'e'; can either be float or None if we want to use argmax
    '''
    if task=='w' or (task=='e' and eps is None):
        Y_pred_argmax = np.argmax(Y_pred, axis=1)
        Y_pred_labels = np.zeros(Y_pred.shape)
        Y_pred_labels[np.arange(Y_pred.shape[0]), Y_pred_argmax] = 1
        Y_pred_labels = Y_pred_labels.astype(int)
    else:
        Y_pred_labels = np.where(Y_pred >= eps, 1, 0)

    return Y_pred_labels

def get_pop_raw(file):
    ''' Function to assign labels based on Eritas's population method.'''
    df =pd.read_csv(join('random_gen', 'data', file))
    df = df.loc[(df['W_min']>= 0) & ((df['Wp_t1'] < 0) | (df['Wp_t2'] < 0) | (df['Wp_t3'] < 0))]
    prob_HandV = abs(0.5*np.ones_like(df['HH']) - (df['HH'] + df['VV']))
    prob_DandA = abs(0.5*np.ones_like(df['HH']) - (df['DD'] + df['AA']))
    prob_RandL = abs(0.5*np.ones_like(df['HH']) - (df['RR'] + df['LL']))
    pop_df = pd.DataFrame()
    pop_df['d_HandV'] = prob_HandV
    pop_df['d_DandA'] = prob_DandA
    pop_df['d_RandL'] = prob_RandL
    return pop_df

def get_labels_pop(file):
    ''' Function to assign labels based on Eritas's population method.'''
    pop_df = get_pop_raw(file)

    # prediction is max value per row
    Y_pred = pop_df[['d_HandV', 'd_DandA', 'd_RandL']].idxmax(axis=1).apply(lambda x: pop_df.columns.get_loc(x))
    Y_pred_labels = np.zeros((len(Y_pred), 3))
    Y_pred_labels[np.arange(len(Y_pred)), Y_pred] = 1
    Y_pred_labels = Y_pred_labels.astype(int)
    return Y_pred_labels

def eval_perf(model, name, file_ls = ['roik_True_400000_r_os_t.csv'], data_ls=None, task='w', input_method='prob_9'):
    ''' Function to measure accuracy on new data from Roik and Matlab. Returns df.
    Params:
        model: ml model object to evaluate
        name: name of model
        file_ls: list of files to evaluate on
        data: list of tuples of X, Y; if not None, use this data instead of loading from file

    '''

    def per_file(file, task, data=None):
        
        if data is None:
            assert file is not None, 'file or data must be provided'
            assert task == 'w' or task =='e', 'task must be w or e'
            X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_method, task=task, split=False)
        else:
            X, Y = data
        if not(name == 'population'):
            Y_pred = model.predict(X)
            Y_pred_labels = get_labels(Y_pred, task)
        else: # population method
            Y_pred_labels = get_labels_pop(file)

        # take dot product of Y and Y_pred_labels to get number of correct predictions
        N_correct = np.sum(np.einsum('ij,ij->i', Y, Y_pred_labels))
        # print(N_correct / len(Y_pred))
        return N_correct / len(Y_pred), N_correct, len(Y_pred)

    # file_ls = ['../../S22_data/all_qual_102_tot.csv']
    p_df = pd.DataFrame()
    if data_ls is None:
        for file in file_ls:
            acc, N_correct, N_total = per_file(file, task)
            p_df = pd.concat([p_df,pd.DataFrame.from_records([{'model':name, 'file':file.split('_')[0], 'acc':acc, 'N_correct':N_correct, 'N_total':N_total}])])
        return p_df
    else:
        for data in data_ls:
            acc, N_correct, N_total = per_file(None, data = data, task=task)
            p_df = pd.concat([p_df,pd.DataFrame.from_records([{'model':name, 'file':'data', 'acc':acc, 'N_correct':N_correct, 'N_total':N_total}])])
        return p_df

def eval_perf_multiple(model_ls, model_names, input_methods, tasks, savename):
    '''Generalizes eval_perf to multiple models and saves to csv.'''
    df = pd.DataFrame()
    for i, model_name in enumerate(list(zip(model_ls, model_names))):
        model = model_name[0]
        name = model_name[1]
        df = pd.concat([df, eval_perf(model, name, task=tasks[i], input_method=input_methods[i])])
    print('saving!')
    df.to_csv(join(new_model_path, savename+'.csv'))

def comp_overlap(model_ls, names, file = 'roik_True_400000_r_os_t.csv', task = 'w', input_method='prob_9'):
    '''Compare overlap of different predictors on 400k test set; computes PCA on inputs and plots overlap of predictions.
    Params:
        model_ls: list of models to compare
        names: list of names of models
        file: file to evaluate on
        task: 'w' or 'e'
        input_method: how to prepare input data; default is 'prob_9' but will modify code to allow a list.
    
    '''
    X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_method, task=task, split=False)

    # create PCA for inputs
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # tsne = TSNE(n_components=2)
    # X_tsne = tsne.fit_transform(X)
    # measure explained variance ratio
    var_ratio = pca.explained_variance_ratio_
    # create fig
    # fig, ax = plt.subplots(1, 2, figsize=(12,7))
    plt.figure(figsize=(10,7))
    plt.title('PCA of Input Probabilities')
    plt.xlabel('PC1: {:.2f}%'.format(var_ratio[0]*100))
    plt.ylabel('PC2: {:.2f}%'.format(var_ratio[1]*100))
    # ax[1].set_xlabel('t-SNE 1')
    # ax[1].set_ylabel('t-SNE 2')
    N_correct_ls = []
    for i, model in enumerate(model_ls):
        # take dot product of Y and Y_pred_labels to get number of correct predictions
        if not(names[i] == 'population'):
            Y_pred = model.predict(X)
            Y_pred_labels = get_labels(Y_pred, task=task)
        else:
            Y_pred_labels = get_labels_pop(file)
        # compute overlap with previous models
        N_correct = np.einsum('ij,ij->i', Y, Y_pred_labels)
        print('-------------------')
        print(names[i]+':')
        if i > 0:
            for j in range(i):
                N_overlap = np.einsum('i,i->', N_correct_ls[j],N_correct)
                print('num overlap with {}: {}'.format(names[j], N_overlap))
                print('perc overlap with {}: {}'.format(names[j], N_overlap/np.sum(N_correct)))
                print('combining models', names[i], 'and', names[j])
                N_sum_vec = N_correct + N_correct_ls[j]
                N_sum = pd.DataFrame()
                N_sum['N_sum'] = N_sum_vec
                N_sum = N_sum.applymap(lambda x: 1 if x > 0 else 0)
                N_sum = np.array(N_sum['N_sum'])
                print('total num correct: {}'.format(np.sum(N_sum)))
                print('total perc correct: {}'.format(np.sum(N_sum)/len(Y)))
        print('-------------------')
        N_correct_ls.append(N_correct)
        # get subset of X where N_correct > 0
        PC1 = X_pca[:, 0][N_correct > 0]
        PC2 = X_pca[:, 1][N_correct > 0]
        plt.scatter(PC1, PC2, label=names[i]+', %.3g'%(np.sum(N_correct) / len(Y)), alpha=0.5)
        # TSNE1 = X_tsne[:, 0][N_correct > 0]
        # TSNE2 = X_tsne[:, 1][N_correct > 0]
        # ax[1].scatter(TSNE1, TSNE2, label=names[i]+', %.3g'%(np.sum(N_correct) / len(Y)), alpha=0.5)
    # ax[0].legend()
    # ax[1].legend()
    # plt.tight_layout()
    plt.legend()
    plt.savefig(join('random_gen', 'models', 'PCA-tsne_comp.pdf'))
    plt.show()

def comp_disagreement(model_ls, names, file = 'roik_True_400000_r_os_t.csv', task = 'w', input_method='prob_9'):
    '''Find the states that disagree between two models and look at confidence levels.'''
    X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_method, task=task, split=False)
    N_correct_ls = []
    for i, model in enumerate(model_ls):
        # take dot product of Y and Y_pred_labels to get number of correct predictions
        if not(names[i] == 'population'):
            Y_pred = model.predict(X)
            Y_pred_labels = get_labels(Y_pred, task=task)
        else:
            Y_pred_labels = get_labels_pop(file)
        # compute overlap with previous models
        N_correct = np.einsum('ij,ij->i', Y, Y_pred_labels)
        N_correct_ls.append(N_correct)
    # look at where model A is correct and model B is incorrect
    A_pred = get_pop_raw(file).to_numpy()
    B_pred = model_ls[1].predict(X)
    
    A_correct_B_incorrect = np.logical_and(N_correct_ls[0] > 0, N_correct_ls[1] == 0)
    # get the confidence levels of model A for these states
    A_confidence = np.max(A_pred[A_correct_B_incorrect], axis=1)
    # get the confidence levels of model B for these states
    B_confidence = np.max(B_pred[A_correct_B_incorrect], axis=1)

    print('A mean', np.mean(A_confidence))
    print('B mean', np.mean(B_confidence))
    print('A sem', np.std(A_confidence) / np.sqrt(len(A_confidence))) 
    print('B sem', np.std(B_confidence) / np.sqrt(len(B_confidence)))

    # do the same for B correct and A incorrect
    B_correct_A_incorrect = np.logical_and(N_correct_ls[1] > 0, N_correct_ls[0] == 0)
    B_confidence2 = np.max(B_pred[B_correct_A_incorrect], axis=1)
    A_confidence2 = np.max(A_pred[B_correct_A_incorrect], axis=1)

    print('A2 mean', np.mean(A_confidence2))
    print('B2 mean', np.mean(B_confidence2))
    print('A2 sem', np.std(A_confidence2) / np.sqrt(len(A_confidence2)))
    print('B2 sem', np.std(B_confidence2) / np.sqrt(len(B_confidence2)))

    # plot the confidence levels
    fig, ax = plt.subplots(1, 2, figsize=(12,7))
    ax[0].hist(A_confidence, bins=20, alpha=0.5, label=names[0])
    ax[0].hist(B_confidence, bins=20, alpha=0.5, label=names[1])
    ax[0].set_xlabel('Confidence')
    ax[0].set_ylabel('Count')
    ax[0].legend()
    ax[0].set_title(f'Model {names[0]} correct, Model {names[1]} incorrect')
    ax[1].hist(A_confidence2, bins=20, alpha=0.5, label=names[0])
    ax[1].hist(B_confidence2, bins=20, alpha=0.5, label=names[1])
    ax[1].set_xlabel('Confidence')
    ax[1].set_ylabel('Count')
    ax[1].legend()
    ax[1].set_title(f'Model {names[1]} correct, Model {names[0]} incorrect')
    plt.tight_layout()
    plt.suptitle('Confidence levels for states where models disagree')
    plt.subplots_adjust(top=0.85)
    plt.savefig(join('random_gen', 'models', 'disagreement_confidence.pdf'))
    plt.show()

if __name__ == '__main__':


    # define models
    new_model_path= join('random_gen', 'models', 'saved_models')
    old_model_path = 'old_models'
    roik_etal_path = join('roik_etal', 'ANN', 'parameters_of_ANN')

    ## load models ##

    # new models ---
    xgb = XGBRegressor()
    xgb.load_model(join(new_model_path, 'r4_s0_0_w_prob_9_xgb_all.json'))
    nn1 = keras.models.load_model(join(new_model_path, 'r4_s0_0_w_prob_9_nn1_all.h5'))
    nn3 = keras.models.load_model(join(new_model_path, 'r4_s0_0_w_prob_9_300_300_300_0.0001_100.h5'))
    nn5 = keras.models.load_model(join(new_model_path, 'r4_s0_6_w_prob_9_300_300_300_300_300_0.0001_100.h5'))

    # # old models ---
    xgb_old = XGBRegressor()
    xgb_old.load_model(join(old_model_path, 'xgbr_best_3.json'))
    json_file = open(join(old_model_path, 'model_qual_v2.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_bl = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model_bl.load_weights(join(old_model_path, "model_qual_v2.h5"))

    ## actual roik et al models ##
    ra_3 = keras.models.load_model(join(roik_etal_path, 'model_3_8333.h5'))
    ra_5 = keras.models.load_model(join(roik_etal_path, 'model_5_8796.h5')) 
    ra_6 = keras.models.load_model(join(roik_etal_path, 'model_6_8946.h5'))
    ra_12 = keras.models.load_model(join(roik_etal_path, 'model_12_937.h5'))
    ra_15 = keras.models.load_model(join(roik_etal_path, 'model_15_9653.h5'))


    ## evaluate ##
    # model_ls = [1, xgb, nn1, nn3, nn5]
    # model_names = ['population', 'xgb', 'nn1', 'nn3', 'nn5']

    # comp_overlap(model_ls, model_names)
    # comp_disagreement(model_ls=[1, nn5], names=['population', 'nn5'])

    model_ls = [ra_3, ra_5, ra_6, ra_12, ra_15]
    model_names = ['ra_3', 'ra_5', 'ra_6', 'ra_12', 'ra_15']
    input_methods = ['prob_3', 'prob_5', 'prob_6', 'prob_12', 'prob_15']
    tasks = ['e' for _ in range(len(model_ls))]
    savename='roik_etal_noeps'

    eval_perf_multiple(model_ls, model_names, input_methods = input_methods, tasks = tasks, savename=savename)

    
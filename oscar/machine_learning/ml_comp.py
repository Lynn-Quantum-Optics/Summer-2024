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

def get_labels_pop(file=None, pop_df=None):
    ''' Function to assign labels based on Eritas's population method.'''
    if pop_df is None and file is not None:
        pop_df = get_pop_raw(file)
    elif pop_df is not None:
        pass

    # prediction is max value per row
    Y_pred = pop_df[['d_HandV', 'd_DandA', 'd_RandL']].idxmax(axis=1).apply(lambda x: pop_df.columns.get_loc(x))
    Y_pred_labels = np.zeros((len(Y_pred), 3))
    Y_pred_labels[np.arange(len(Y_pred)), Y_pred] = 1
    Y_pred_labels = Y_pred_labels.astype(int)
    return Y_pred_labels

def eval_perf(model, name, file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_9', pop_method='none', normalize=False, conc_threshold=0):
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
            X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_method, pop_method = pop_method, task=task, split=False, normalize=normalize, conc_threshold=conc_threshold)
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
        return N_correct / len(Y_pred_labels), N_correct, len(Y_pred_labels)

    # file_ls = ['../../S22_data/all_qual_102_tot.csv']
    p_df = pd.DataFrame()
    if data_ls is None:
        for i, file in enumerate(file_ls):
            acc, N_correct, N_total = per_file(file, task)
            p_df = pd.concat([p_df,pd.DataFrame.from_records([{'model':name, 'file':file_names[i], 'acc':acc, 'N_correct':N_correct, 'N_total':N_total, 'conc_threshold':conc_threshold}])])
        return p_df
    else:
        for data in data_ls:
            acc, N_correct, N_total = per_file(None, data = data, task=task)
            p_df = pd.concat([p_df,pd.DataFrame.from_records([{'model':name, 'file':'data', 'acc':acc, 'N_correct':N_correct, 'N_total':N_total, 'conc_threshold':conc_threshold}])])
        return p_df

def eval_perf_multiple(model_ls, model_names, input_methods, pop_methods, tasks, savename, file_ls = ['roik_True_400000_r_os_t.csv'], file_names=['Test']):
    '''Generalizes eval_perf to multiple models and saves to csv.'''
    df = pd.DataFrame()
    for i, model_name in enumerate(list(zip(model_ls, model_names))):
        model = model_name[0]
        name = model_name[1]
        df = pd.concat([df, eval_perf(model, name, task=tasks[i], input_method=input_methods[i], pop_method=pop_methods[i], file_ls=file_ls, file_names=file_names)])
    print('saving!')
    df.to_csv(join(new_model_path, savename+'.csv'))

def comp_overlap(model_ls, names, input_methods, pop_methods, file = 'roik_400k_extra_wpop_rd.csv', task = 'w'):
    '''Compare overlap of different predictors on 400k test set; computes PCA on inputs and plots overlap of predictions.
    Params:
        model_ls: list of models to compare
        names: list of names of models
        file: file to evaluate on
        task: 'w' or 'e'
        input_method: how to prepare input data; default is 'prob_9' but will modify code to allow a list.
    
    '''
    # # create PCA for inputs
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)

    # # tsne = TSNE(n_components=2)
    # # X_tsne = tsne.fit_transform(X)
    # # measure explained variance ratio
    # var_ratio = pca.explained_variance_ratio_
    # # create fig
    # # fig, ax = plt.subplots(1, 2, figsize=(12,7))
    # plt.figure(figsize=(10,7))
    # plt.title('PCA of Input Probabilities')
    # plt.xlabel('PC1: {:.2f}%'.format(var_ratio[0]*100))
    # plt.ylabel('PC2: {:.2f}%'.format(var_ratio[1]*100))
    # ax[1].set_xlabel('t-SNE 1')
    # ax[1].set_ylabel('t-SNE 2')
    N_correct_ls = []
    for i, model in enumerate(model_ls):
        X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_methods[i], pop_method = pop_methods[i], task=task, split=False)

        # take dot product of Y and Y_pred_labels to get number of correct predictions
        if not(names[i] == 'population'):
            Y_pred = model.predict(X)
            Y_pred_labels = get_labels(Y_pred, task=task)
            print(input_methods[i], pop_methods[i])
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
        # # get subset of X where N_correct > 0
        # PC1 = X_pca[:, 0][N_correct > 0]
        # PC2 = X_pca[:, 1][N_correct > 0]
        # plt.scatter(PC1, PC2, label=names[i]+', %.3g'%(np.sum(N_correct) / len(Y)), alpha=0.5)
        # TSNE1 = X_tsne[:, 0][N_correct > 0]
        # TSNE2 = X_tsne[:, 1][N_correct > 0]
        # ax[1].scatter(TSNE1, TSNE2, label=names[i]+', %.3g'%(np.sum(N_correct) / len(Y)), alpha=0.5)
    # ax[0].legend()
    # ax[1].legend()
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(join('random_gen', 'models', 'PCA-tsne_comp.pdf'))
    # plt.show()

def comp_disagreement(model_ls, names, file = 'roik_True_400000_r_os_t.csv', task = 'w', input_method='prob_9'):
    '''Find the states that disagree between two models and look at confidence levels.
    '''
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

def det_threshold(nn5, file = 'roik_True_800000_r_os_v.csv', task = 'w', input_method='prob_9'):
    '''Look at the distribution of confidence levels for all states between the models on the validation data.'''
    X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_method, pop_method='none', task=task, split=False)
    # get predictions for each
    Y_pred_nn5 = nn5.predict(X)
    confidence_nn5 = np.max(Y_pred_nn5, axis=1)
    Y_pred_nn5_labels = get_labels(Y_pred_nn5, task=task)
    Y_pred_nn5_ncorrect = np.einsum('ij,ij->i', Y, Y_pred_nn5_labels)
    Y_pred_pop = get_pop_raw(file=file)
    confidence_pop = np.max(Y_pred_pop.to_numpy(), axis=1)
    # get difference bewtween max and next max for pop
    pop_diff_max = np.max(Y_pred_pop.to_numpy(), axis=1) - np.partition(Y_pred_pop.to_numpy(), -2, axis=1)[:,-2] # partition is faster than argsort; -2 is second largest, orders from smallest to largest
    # get difference between max  and second next max for pop
    pop_diff_max2 = np.max(Y_pred_pop.to_numpy(), axis=1) - np.partition(Y_pred_pop.to_numpy(), -3, axis=1)[:,-3]
  
    Y_pred_pop_labels = get_labels_pop(pop_df = Y_pred_pop)
    Y_pred_pop_ncorrect = np.einsum('ij,ij->i', Y, Y_pred_pop_labels)

    pop_diff_max_correct = pop_diff_max[Y_pred_pop_ncorrect > 0]
    pop_diff_max_incorrect = pop_diff_max[Y_pred_pop_ncorrect == 0]
    pop_diff_max2_correct = pop_diff_max2[Y_pred_pop_ncorrect > 0]
    pop_diff_max2_incorrect = pop_diff_max2[Y_pred_pop_ncorrect == 0]

    # plot the confidence levels, with correct states in green and incorrect in red
    fig, ax = plt.subplots(3, 2, figsize=(30,15))
    confidence_nn5_correct = confidence_nn5[Y_pred_nn5_ncorrect > 0]
    confidence_nn5_incorrect = confidence_nn5[Y_pred_nn5_ncorrect == 0]
    ax[0,0].hist(confidence_nn5_correct, bins=20, alpha=0.5, label='correct', color='green')
    ax[0,0].hist(confidence_nn5_incorrect, bins=20, alpha=0.5, label='incorrect', color='red')
    ax[0,0].set_xlabel('Confidence')
    ax[0,0].set_ylabel('Count')
    ax[0,0].legend()
    ax[0,0].set_title(f'NN5')
    # replot, but with values <= .6
    # nc, binc, _ = ax[1,0].hist(confidence_nn5_correct[confidence_nn5_correct <= .6], bins=5, alpha=0.5, label='correct', color='green')
    # ni, bini, _ = ax[1,0].hist(confidence_nn5_incorrect[confidence_nn5_incorrect <= 0.6], bins=5, alpha=0.5, label='incorrect', color='red')
    # # annotate with counts
    # for i in range(len(nc)):
    #     ax[1,0].annotate(str(int(nc[i])), xy=(binc[i], nc[i]), xytext=(binc[i]+2e-2, nc[i]+10), ha='center', va='center')
    #     ax[1,0].annotate(str(int(ni[i])), xy=(bini[i], ni[i]), xytext=(bini[i]+2e-2, ni[i]+10), ha='center', va='center')
    # ax[1,0].set_xlabel('Confidence')
    # ax[1,0].set_ylabel('Count')
    # ax[1,0].legend()
    # ax[1,0].set_title(f'NN5')

    confidence_pop_correct = confidence_pop[Y_pred_pop_ncorrect > 0]
    confidence_pop_incorrect = confidence_pop[Y_pred_pop_ncorrect == 0]
    ax[0,1].hist(confidence_pop_correct, bins=20, alpha=0.5, label='correct', color='green')
    ax[0,1].hist(confidence_pop_incorrect, bins=20, alpha=0.5, label='incorrect', color='red')
    ax[0,1].set_xlabel('Confidence')
    ax[0,1].set_ylabel('Count')
    ax[0,1].legend()
    ax[0,1].set_title(f'Population')
    # nc, binc, _ = ax[1,1].hist(confidence_pop_correct[confidence_pop_correct >= .4], bins=5, alpha=0.5, label='correct', color='green')
    # ni, bini, _ = ax[1,1].hist(confidence_pop_incorrect[confidence_pop_incorrect >= .4], bins=5, alpha=0.5, label='incorrect', color='red')
    # # annotate with counts
    # for i in range(len(nc)):
    #     ax[1,1].annotate(f'{nc[i]}', ((binc[i]+binc[i+1])/2, nc[i]), ha='center', va='bottom')
    #     ax[1,1].annotate(f'{ni[i]}', ((bini[i]+bini[i+1])/2, ni[i]), ha='center', va='bottom')
    # ax[1,1].set_xlabel('Confidence')
    # ax[1,1].set_ylabel('Count')
    # ax[1,1].legend()
    # ax[1,1].set_title(f'Population')

    # ax[2,0].hist(pop_diff_max_correct, bins=20, alpha=0.5, label='max - next max', color='blue')
    # ax[2,0].hist(pop_diff_max2_correct, bins=20, alpha=0.5, label='max - second next max', color='red')
    # ax[2,0].set_xlabel('Confidence')
    # ax[2,0].set_ylabel('Count')
    # ax[2,0].legend()
    # ax[2,0].set_title(f'Population Differences, Correct')
    # nmax, binmax, _ = ax[2,1].hist(pop_diff_max_correct[pop_diff_max_correct >= .4], bins=5, alpha=0.5, label='max - next max', color='blue')
    # nmax2, binmax2, _ = ax[2,1].hist(pop_diff_max2_correct[pop_diff_max2_correct >= .4], bins=5, alpha=0.5, label='max - second next max', color='red')
    # # annotate with counts
    # for i in range(len(nmax)):
    #     ax[2,1].annotate(f'{nmax[i]}', ((binmax[i]+binmax[i+1])/2, nmax[i]), ha='center', va='bottom')
    #     ax[2,1].annotate(f'{nmax2[i]}', ((binmax2[i]+binmax2[i+1])/2, nmax2[i]), ha='center', va='bottom')
    # ax[2,1].set_xlabel('Confidence')
    # ax[2,1].set_ylabel('Count')
    # ax[2,1].legend()
    # ax[2,1].set_title(f'Population Differences Correct, zoomed')

    # ax[3,0].hist(pop_diff_max_incorrect, bins=20, alpha=0.5, label='max - next max', color='blue')
    # ax[3,0].hist(pop_diff_max2_incorrect, bins=20, alpha=0.5, label='max - second next max', color='red')
    # ax[3,0].set_xlabel('Confidence')
    # ax[3,0].set_ylabel('Count')
    # ax[3,0].legend()
    # ax[3,0].set_title(f'Population Differences, Incorrect')
    # nmax, binmax, _ = ax[3,1].hist(pop_diff_max_incorrect[pop_diff_max_incorrect >= .4], bins=5, alpha=0.5, label='max - next max', color='blue')
    # nmax2, binmax2, _ = ax[3,1].hist(pop_diff_max2_incorrect[pop_diff_max2_incorrect >= .4], bins=5, alpha=0.5, label='max - second next max', color='red')
    # # annotate with counts
    # for i in range(len(nmax)):
    #     ax[3,1].annotate(f'{nmax[i]}', ((binmax[i]+binmax[i+1])/2, nmax[i]), ha='center', va='bottom')
    #     ax[3,1].annotate(f'{nmax2[i]}', ((binmax2[i]+binmax2[i+1])/2, nmax2[i]), ha='center', va='bottom')
    # ax[3,1].set_xlabel('Confidence')
    # ax[3,1].set_ylabel('Count')
    # ax[3,1].legend()
    # ax[3,1].set_title(f'Population Differences Incorrect, zoomed')

    # plot pop diff max and max2 where nn5 is correct and pop is wrong; and vice versa
    ax[1,0].hist(pop_diff_max[(Y_pred_nn5_ncorrect > 0) & (Y_pred_pop_ncorrect == 0)], bins=20, alpha=0.5, label='max - next max', color='blue')
    ax[1,0].hist(pop_diff_max2[(Y_pred_nn5_ncorrect > 0) & (Y_pred_pop_ncorrect == 0)], bins=20, alpha=0.5, label='max - second next max', color='red')
    ax[1,0].set_xlabel('Confidence')
    ax[1,0].set_ylabel('Count')
    ax[1,0].legend()
    ax[1,0].set_title(f'Population Differences, NN5 Correct, Population Incorrect')
    
    ax[1,1].hist(pop_diff_max[(Y_pred_nn5_ncorrect == 0) & (Y_pred_pop_ncorrect > 0)], bins=20, alpha=0.5, label='max - next max', color='blue')
    ax[1,1].hist(pop_diff_max2[(Y_pred_nn5_ncorrect == 0) & (Y_pred_pop_ncorrect > 0)], bins=20, alpha=0.5, label='max - second next max', color='red')
    ax[1,1].set_xlabel('Confidence')
    ax[1,1].set_ylabel('Count')
    ax[1,1].legend()
    ax[1,1].set_title(f'Population Differences, NN5 Incorrect, Population Correct')

    # plot nn5 confidence where pop is correct and nn5 is wrong; and vice versa
    ax[2,0].hist(confidence_nn5[(Y_pred_nn5_ncorrect == 0) & (Y_pred_pop_ncorrect > 0)], bins=20, alpha=0.5, label='NN5', color='blue')
    ax[2,0].set_xlabel('Confidence')
    ax[2,0].set_ylabel('Count')
    ax[2,0].legend()
    ax[2,0].set_title(f'NN5 Confidence, NN5 Incorrect, Population Correct')
    ax[2,1].hist(confidence_nn5[(Y_pred_nn5_ncorrect > 0) & (Y_pred_pop_ncorrect == 0)], bins=20, alpha=0.5, label='NN5', color='blue')
    ax[2,1].set_xlabel('Confidence')
    ax[2,1].set_ylabel('Count')
    ax[2,1].legend()
    ax[2,1].set_title(f'NN5 Confidence, NN5 Correct, Population Incorrect')


    plt.tight_layout()
    plt.suptitle('Confidence levels on Validation Data')
    plt.subplots_adjust(top=0.95)
    plt.savefig(join('random_gen', 'models', 'det_threshold.pdf'))
    # plt.show()
    plt.close()

def det_threshold_gd(file = 'roik_True_4000000_r_os_v.csv', task = 'w', input_method='prob_9'):
    '''Use gradient descent to optimize parameters on how we choose the threshold for combining nn5 and pop'''
    X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_method, pop_method='none', task=task, split=False)
    # get predictions for each
    Y_pred_nn5 = nn5.predict(X)
    Y_pred_pop = get_pop_raw(file=file)
    
def plot_comp_acc(steps=50, include_all=False):
    '''Plot accuracy as we vary conc_threshold for both nn5 and pop as well as just W and W' full'''
    # get data
    conc_threshold_ls = np.linspace(0, .12, steps)
    nn5_acc = []
    pop_acc = []
    bl_acc = []
    xgb_old_acc = []
    xgb_new_acc = []
    nn1_acc = []
    nn3_acc = []
    nn5 = keras.models.load_model(join('random_gen', 'models', 'saved_models', 'r4_s0_6_w_prob_9_300_300_300_300_300_0.0001_100.h5'))
    xgb_new = XGBRegressor()
    xgb_new.load_model(join('random_gen', 'models', 'r4_s0_0_w_prob_9_xgb_all.json'))
    nn1 = keras.models.load_model(join('random_gen', 'models', 'saved_models', 'r4_s0_0_w_prob_9_300_0.0001_100.h5'))
    nn3 = keras.models.load_model(join('random_gen', 'models', 'saved_models', 'r4_s0_0_w_prob_9_300_300_300_0.0001_100.h5'))
    xgb_old = XGBRegressor()
    xgb_old.load_model(join(old_model_path, 'xgbr_best_3.json'))
    json_file = open(join(old_model_path, 'model_qual_v2.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_bl = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model_bl.load_weights(join(old_model_path, "model_qual_v2.h5"))
    for conc_threshold in conc_threshold_ls:
        nn5_acc.append(eval_perf(nn5, 'nn5', file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_9', pop_method='none', normalize=False, conc_threshold=conc_threshold)['acc'].values[0])

        pop_acc.append(eval_perf(1, 'population', file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_9', pop_method='none', normalize=False, conc_threshold=conc_threshold)['acc'].values[0])

        bl_acc.append(eval_perf(model_bl, 'bl', file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_12_red', pop_method='none', normalize=False, conc_threshold=conc_threshold)['acc'].values[0])

        if include_all:

            xgb_old_acc.append(eval_perf(xgb_old, 'xgb_old', file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_12_red', pop_method='none', normalize=False, conc_threshold=conc_threshold)['acc'].values[0])

            xgb_new_acc.append(eval_perf(xgb_new, 'xgb_new', file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_9', pop_method='none', normalize=False, conc_threshold=conc_threshold)['acc'].values[0])

            nn1_acc.append(eval_perf(nn1, 'nn1', file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_9', pop_method='none', normalize=False, conc_threshold=conc_threshold)['acc'].values[0])

            nn3_acc.append(eval_perf(nn3, 'nn3', file_ls = ['roik_True_400000_r_os_t.csv'], file_names = ['Test'], data_ls=None, task='w', input_method='prob_9', pop_method='none', normalize=False, conc_threshold=conc_threshold)['acc'].values[0])

    # plot
    plt.figure(figsize=(10,7))
    plt.plot(conc_threshold_ls, nn5_acc, label='NN5')
    plt.plot(conc_threshold_ls, pop_acc, label='Population')
    plt.plot(conc_threshold_ls, bl_acc, label='NN3, prev')
    if include_all:
        plt.plot(conc_threshold_ls, xgb_old_acc, label='XGB, 0')
        plt.plot(conc_threshold_ls, xgb_new_acc, label='XGB, 1')
        plt.plot(conc_threshold_ls, nn1_acc, label='NN1')
        plt.plot(conc_threshold_ls, nn3_acc, label='NN3')
    plt.xlabel('Concurrence Threshold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Comparison of $W'$ Model Performance")
    plt.savefig(join('random_gen', 'models', f'comp_acc_{steps}_{include_all}.pdf'))

def display_model(model):
    '''Display keras model architecture
    '''
    from keras.utils.vis_utils import plot_model

    plot_model(model, to_file=join('random_gen', 'models', 'model_plot.pdf'), show_shapes=True, show_layer_names=True)

    # plot sigmoid and relu
    x = np.linspace(-10, 10, 100)
    plt.plot(x, 1/(1+np.exp(-x)))
    plt.title('Sigmoid Activation Function, $S(x) = (1+e^{-x})^{-1}$')
    plt.savefig(join('random_gen', 'models', 'sigmoid.pdf'))
    plt.clf()
    plt.plot(x, np.maximum(x, 0))
    plt.title('ReLU Activation Function, $R(x) = max(0, x)$')
    plt.savefig(join('random_gen', 'models', 'relu.pdf'))


    

if __name__ == '__main__':


    # define models
    new_model_path= join('random_gen', 'models', 'saved_models')
    old_model_path = 'old_models'
    roik_etal_path = join('roik_etal', 'ANN', 'parameters_of_ANN')

    ## load models ##

    # new models ---
    # xgb = XGBRegressor()
    # xgb.load_model(join(new_model_path, 'r4_s0_0_w_prob_9_xgb_all.json'))
    # nn1 = keras.models.load_model(join(new_model_path, 'r4_s0_0_w_prob_9_nn1_all.h5'))
    # nn3 = keras.models.load_model(join(new_model_path, 'r4_s0_0_w_prob_9_300_300_300_0.0001_100.h5'))
    nn5 = keras.models.load_model(join(new_model_path, 'r4_s0_6_w_prob_9_300_300_300_300_300_0.0001_100.h5'))
    nn5_wraw = keras.models.load_model(join(new_model_path, 'r4_s0_47_w_prob_9_raw_200_200_200_200_200_0.0001_100.h5'))
    nn5_wdiff = keras.models.load_model(join(new_model_path, 'r4_s0_48_w_prob_9_diff_200_200_200_200_200_0.0001_100.h5'))
    nn5_wrd = keras.models.load_model(join(new_model_path, 'r4_s0_49_w_prob_9_rd_200_200_200_200_200_0.0001_100.h5'))
    nn5_rawonly = keras.models.load_model(join(new_model_path, 'r4_s0_50_w_none_raw_200_200_200_200_200_0.0001_100.h5'))
    nn5_diffonly = keras.models.load_model(join(new_model_path, 'r4_s0_51_w_none_diff_200_200_200_200_200_0.0001_100.h5'))
    nn5_rdonly = keras.models.load_model(join(new_model_path, 'r4_s0_52_w_none_rd_200_200_200_200_200_0.0001_100.h5'))

    model_ls = [1, nn5, nn5_wraw, nn5_wdiff, nn5_wrd, nn5_rawonly, nn5_diffonly, nn5_rdonly]
    model_names = ['population', 'nn5', 'nn5_wraw', 'nn5_wdiff', 'nn5_wrd', 'nn5_rawonly', 'nn5_diffonly', 'nn5_rdonly']
    input_methods = ['prob_9', 'prob_9', 'prob_9', 'prob_9', 'prob_9', 'none', 'none', 'none']
    pop_methods = ['none', 'none', 'raw', 'diff', 'rd', 'raw', 'diff', 'rd']
    # comp_overlap(model_ls=model_ls, names=model_names, input_methods=input_methods, pop_methods=pop_methods)

    # # old models ---
    # xgb_old = XGBRegressor()
    # xgb_old.load_model(join(old_model_path, 'xgbr_best_3.json'))
    # json_file = open(join(old_model_path, 'model_qual_v2.json'), 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model_bl = keras.models.model_from_json(loaded_model_json)
    # # load weights into new model
    # model_bl.load_weights(join(old_model_path, "model_qual_v2.h5"))

    # ## actual roik et al models ##
    # ra_3 = keras.models.load_model(join(roik_etal_path, 'model_3_8333.h5'))
    # ra_5 = keras.models.load_model(join(roik_etal_path, 'model_5_8796.h5')) 
    # ra_6 = keras.models.load_model(join(roik_etal_path, 'model_6_8946.h5'))
    # ra_12 = keras.models.load_model(join(roik_etal_path, 'model_12_937.h5'))
    # ra_15 = keras.models.load_model(join(roik_etal_path, 'model_15_9653.h5'))

    ## evaluate ##
    # model_ls = [1, xgb, nn1, nn3, nn5]
    # model_names = ['population', 'xgb', 'nn1', 'nn3', 'nn5']

    # comp_overlap(model_ls, model_names)
    # comp_disagreement(model_ls=[1, nn5], names=['population', 'nn5'])

    # model_ls = [ra_3, ra_5, ra_6, ra_12, ra_15]
    # model_names = ['ra_3', 'ra_5', 'ra_6', 'ra_12', 'ra_15']
    # input_methods = ['prob_3_r', 'prob_5_r', 'prob_6_r', 'prob_12_r', 'prob_15_r']
    # pop_methods = ['none' for _ in range(len(model_ls))]
    # tasks = ['e' for _ in range(len(model_ls))]
    # savename='roik_etal_noeps'

    # eval_perf_multiple(model_ls, model_names, input_methods = input_methods, pop_methods = pop_methods, tasks = tasks, savename=savename, file_ls = ['roik_True_400000_w_roik.csv'], file_names=['roik_400k'])

    det_threshold(nn5)
    # print('nn5')
    # plot_comp_acc(include_all=True)
    # display_model(nn5)

    
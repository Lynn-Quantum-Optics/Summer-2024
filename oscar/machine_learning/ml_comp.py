# file to compute performance of models on new data, specifically 400k Roik and 700k Matlab
from os.path import join
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import keras

from multiprocessing import Pool, cpu_count

from train_prep import prepare_data
from rho_methods import compute_witnesses

def eval_perf(model, name, file_ls = ['roik_True_400000_r_os_t.csv'], data_ls=None, task='w', input_method='prob_9'):
    ''' Function to measure accuracy on new data from Roik and Matlab. Returns df.
    Params:
        model: ml model object to evaluate
        name: name of model
        file_ls: list of files to evaluate on
        data: list of tuples of X, Y; if not None, use this data instead of loading from file

    '''
    def get_labels(Y_pred):
        ''' Function to assign labels based on argmax per row'''
        Y_pred_argmax = np.argmax(Y_pred, axis=1)
        Y_pred_labels = np.zeros(Y_pred.shape)
        Y_pred_labels[np.arange(Y_pred.shape[0]), Y_pred_argmax] = 1
        Y_pred_labels = Y_pred_labels.astype(int)
        return Y_pred_labels

    def per_file(file, data=None):
        
        if data is None:
            assert file is not None, 'file or data must be provided'
            assert task == 'w' or task =='e', 'task must be w or e'
            X, Y = prepare_data(join('random_gen', 'data'), file, input_method=input_method, task=task, split=False)
        else:
            X, Y = data
        if not(name == 'population'):
            Y_pred = model.predict(X)
            Y_pred_labels = get_labels(Y_pred)
        else: # population method
            df =pd.read_csv(join('random_gen', 'data', file))
            df = df.loc[(df['W_min']>= 0) & ((df['Wp_t1'] < 0) | (df['Wp_t2'] < 0) | (df['Wp_t3'] < 0))]
            prob_HandV = abs(0.5*np.ones_like(df['HH']) - (df['HH'] + df['VV']))
            prob_DandA = abs(0.5*np.ones_like(df['HH']) - (df['DD'] + df['AA']))
            prob_RandL = abs(0.5*np.ones_like(df['HH']) - (df['RR'] + df['LL']))
            pop_df = pd.DataFrame()
            pop_df['d_HandV'] = prob_HandV
            pop_df['d_DandA'] = prob_DandA
            pop_df['d_RandL'] = prob_RandL

            # prediction is max value per row
            Y_pred = pop_df[['d_HandV', 'd_DandA', 'd_RandL']].idxmax(axis=1).apply(lambda x: pop_df.columns.get_loc(x))
            Y_pred_labels = np.zeros((len(Y_pred), 3))
            Y_pred_labels[np.arange(len(Y_pred)), Y_pred] = 1
            Y_pred_labels = Y_pred_labels.astype(int)

        # take dot product of Y and Y_pred_labels to get number of correct predictions
        N_correct = np.sum(np.einsum('ij,ij->i', Y, Y_pred_labels))
        # print(N_correct / len(Y_pred))
        return N_correct / len(Y_pred), N_correct, len(Y_pred)

    # file_ls = ['../../S22_data/all_qual_102_tot.csv']
    p_df = pd.DataFrame()
    if data_ls is None:
        for file in file_ls:
            acc, N_correct, N_total = per_file(file)
            p_df = pd.concat([p_df,pd.DataFrame.from_records([{'model':name, 'file':file.split('_')[0], 'acc':acc, 'N_correct':N_correct, 'N_total':N_total}])])
        return p_df
    else:
        for data in data_ls:
            acc, N_correct, N_total = per_file(None, data = data)
            p_df = pd.concat([p_df,pd.DataFrame.from_records([{'model':name, 'file':'data', 'acc':acc, 'N_correct':N_correct, 'N_total':N_total}])])
        return p_df

if __name__ == '__main__':


    # define models
    new_model_path= join('random_gen', 'models', 'w_7_12')
    old_model_path = 'old_models'

    ## load models ##

    # new models ---
    xgb = XGBRegressor()
    xgb.load_model(join(new_model_path, 'xgb_drawn9.json'))
    nn1 = keras.models.load_model(join(new_model_path, 'nn1_valiant29.h5'))
    nn3 = keras.models.load_model(join(new_model_path, 'nn3_woven66.h5'))
    nn5 = keras.models.load_model(join(new_model_path, 'nn5_polar34.h5'))

    # # old models ---
    xgb_old = XGBRegressor()
    xgb_old.load_model(join(old_model_path, 'xgbr_best_3.json'))
    json_file = open(join(old_model_path, 'model_qual_v2.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_bl = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model_bl.load_weights(join(old_model_path, "model_qual_v2.h5"))


    ## evaluate ##
    model_ls = [xgb, nn1, nn3, nn5, xgb_old, model_bl]
    model_names = ['xgb', 'nn1', 'nn3', 'nn5', 'xgb_old', 'bl_old']
    # model_ls = [1]
    # model_names = ['population']
    # eval_perf(1, 'population', file_ls = ['roik_True_400000_r_os_t.csv'])
    df = pd.DataFrame()
    for model, name in zip(model_ls, model_names):
        df = pd.concat([df, eval_perf(model, name)])
    print('saving!')
    df.to_csv(join(new_model_path, 'model_perf.csv'))
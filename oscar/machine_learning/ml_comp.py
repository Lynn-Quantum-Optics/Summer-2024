# file to compute performance of models on new data, specifically 400k Roik and 700k Matlab
from os.path import join
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import keras

from multiprocessing import Pool, cpu_count

from train_prep import prepare_data
from rho_methods import compute_witnesses

def eval_perf(model, name):
    ''' Function to measure accuracy on new data from Roik and Matlab. Returns df.'''
    def get_labels(Y_pred):
        ''' Function to assign labels based on argmax per row'''
        Y_pred_argmax = np.argmax(Y_pred, axis=1)
        Y_pred_labels = np.zeros(Y_pred.shape)
        Y_pred_labels[np.arange(Y_pred.shape[0]), Y_pred_argmax] = 1
        Y_pred_labels = Y_pred_labels.astype(int)
        return Y_pred_labels

    
    def per_file(file):
        if not('old' in name):
            X, Y = prepare_data(join('random_gen', 'data'), file, input_method='prob_9', task='w', split=False)
        else:
            X, Y = prepare_data(join('random_gen', 'data'), file, input_method='prob_12_red', task='w', split=False)
        if not(name == 'population'):
            Y_pred = model.predict(X)
            Y_pred_labels = get_labels(Y_pred)
        else: # population method
            prob_HandV = abs(0.5*np.ones_like(df['HH']) - (df['HH'] + df['VV']))
            prob_DandA = abs(0.5*np.ones_like(df['HH']) - (df['DD'] + df['AA']))
            prob_RandL = abs(0.5*np.ones_like(df['HH']) - (df['RR'] + df['LL']))
            pop_df = pd.DataFrame()
            pop_df['d_HandV'] = prob_HandV
            pop_df['d_DandA'] = prob_DandA
            pop_df['d_RandL'] = prob_RandL

            # prediction is max value per row
            Y_pred_labels = df[['d_HandV', 'd_DandA', 'd_RandL']].idxmax(axis=1)
        # take dot product of Y and Y_pred_labels to get number of correct predictions
        N_correct = np.sum(np.einsum('ij,ij->i', Y, Y_pred_labels))
        print(N_correct / len(Y_pred))
        return N_correct / len(Y_pred), N_correct, len(Y_pred)

    file_ls = ['roik_True_400000_t.csv', 'm_total.csv']
    # file_ls = ['../../S22_data/all_qual_102_tot.csv']
    p_df = pd.DataFrame()
    for file in file_ls:
        acc, N_correct, N_total = per_file(file)
        p_df = pd.concat([p_df,pd.DataFrame.from_records([{'model':name, 'file':file.split('_')[0], 'acc':acc, 'N_correct':N_correct, 'N_total':N_total}])])
    return p_df


# define models
new_model_path= join('random_gen', 'models', 'w_7_7')
old_model_path = 'old_models'

## load models ##

# new models ---
xgb = XGBRegressor()
xgb.load_model(join(new_model_path, 'xgb_w_prob_9_r4.json'))

# old models ---
xgb_old = XGBRegressor()
xgb_old.load_model(join(old_model_path, 'xgbr_best_3.json'))

json_file = open(join(old_model_path, 'model_qual_v2.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model_bl = keras.models.model_from_json(loaded_model_json)
# load weights into new model
model_bl.load_weights(join(old_model_path, "model_qual_v2.h5"))


## evaluate ##
model_ls = [xgb, xgb_old, model_bl]
model_names = ['xgb', 'xgb_old', 'bl_old']
df = pd.DataFrame()
for model, name in zip(model_ls, model_names):
    df = pd.concat([df, eval_perf(model, name)])
print('saving!')
df.to_csv(join(new_model_path, 'model_perf.csv'))
# file to check performance of BL model on new data and new model on old data
import keras
from xgboost import XGBRegressor
from ml_evaluate import *
from os.path import join

def get_labels(Y_pred):
    ''' Function to assign labels based on argmax per row'''
    Y_pred_argmax = np.argmax(Y_pred, axis=1)
    Y_pred_labels = np.zeros(Y_pred.shape)
    Y_pred_labels[np.arange(Y_pred.shape[0]), Y_pred_argmax] = 1
    Y_pred_labels = Y_pred_labels.astype(int)
    return Y_pred_labels

def test_bl():

    # read in model
    json_file = open('models/model_qual_v2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_bl = keras.models.model_from_json(loaded_model_json)

    DATA_PATH = 'random_gen/data'

    # calculate performance on all files
    files = ['hurwitz_True_4400000_b0_method_0.csv', 'hurwitz_True_4400000_b0_method_1.csv', 'hurwitz_True_4400000_b0_method_2.csv']

    file_names = ['method_0', 'method_1', 'method_2']

    acc_df= pd.DataFrame(columns=['file', 'acc', 'N_correct', 'N_total'])
    for i, file in enumerate(files):
        df = pd.read_csv(join(DATA_PATH, file))
        df = df.loc[(df['W_min']>=0) & ((df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t3']<0))].reset_index()

        inputs= ['HH', 'HV', 'VH', 'VV', 'DD', 'DA', 'AD', 'AA', 'RR', 'RL', 'LR', 'LL']
        outputs =['Wp_t1', 'Wp_t2', 'Wp_t3']

        X = df[inputs].to_numpy()
        Y = df[outputs]
        Y = Y.applymap(lambda x: 1 if x < 0 else 0)
        Y = Y.to_numpy()
        print(Y)

        # evaluate model
        print('Evaluating model on file: ', file)
        Y_pred = model_bl.predict(X)
        Y_pred_labels = get_labels(Y_pred)
        print(Y_pred_labels)

        N_correct = np.sum(np.einsum('ij,ij->i', Y, Y_pred_labels))
        acc = N_correct / len(Y_pred)
        acc_df = pd.concat([acc_df, pd.DataFrame.from_records([{'file':file_names[i], 'acc':acc, 'N_correct':N_correct, 'N_total':len(Y_pred)}])])

    acc_df.to_csv('models/acc_bl.csv', index=False)

def test_new():
    xgb = XGBRegressor()
    xgb.load_model('radom_gen/models/prob_h0/xgb_w_prob_9_h0.json')

    file = 'S22_data/all_qual_20000_5.csv'
    df = pd.read_csv(join(DATA_PATH, file))
    df = df.loc[(df['W_min']>=0) & ((df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t3']<0))].reset_index()

    inputs= ['HH', 'HV', 'VV', 'DD', 'DA', 'AA', 'RR', 'RL', 'LL']
    outputs =['Wp_t1', 'Wp_t2', 'Wp_t3']

    X = df[inputs].to_numpy()
    Y = df[outputs]
    Y = Y.applymap(lambda x: 1 if x < 0 else 0)
    Y = Y.to_numpy()
    print(Y)

    # evaluate model
    print('Evaluating model on file: ', file)
    Y_pred = xgb.predict(X)
    Y_pred_labels = get_labels(Y_pred)
    print(Y_pred_labels)

    N_correct = np.sum(np.einsum('ij,ij->i', Y, Y_pred_labels))
    acc = N_correct / len(Y_pred)
    print('acc', acc)
    print('N_correct', N_correct)
    print('size', len(Y_pred))
        



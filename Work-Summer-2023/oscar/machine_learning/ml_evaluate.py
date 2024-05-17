# file to evaluate performance of machine learning models
from os.path import join
import numpy as np
import pandas as pd

from train_prep import prepare_data

def evaluate_perf(model,split, datapath, file, input_method, task, p=0.8):
    ''' Function to measure accuracy on both train and test data.
    Params:
        model: trained model
        split: boolean for whether to split data into train and test
        datapath: path to csv
        savename: what to name the prepped data
        input_method: how to prepare the input data: e.g., diag for XX, YY, ZZ
        task: 'witness' or 'entangled' for prediction method
        threshold: value above which to be considered 1
        p: fraction of data to use for training
        '''
    def get_labels(Y_pred):
        ''' Function to assign labels based on argmax per row'''
        Y_pred_argmax = np.argmax(Y_pred, axis=1)
        Y_pred_labels = np.zeros(Y_pred.shape)
        Y_pred_labels[np.arange(Y_pred.shape[0]), Y_pred_argmax] = 1
        Y_pred_labels = Y_pred_labels.astype(int)
        return Y_pred_labels

    print(split)

    if split: # if checking performance on the data we used to train the model
        X_train, Y_train, X_test, Y_test = prepare_data(datapath, file, input_method, task, split, p)
        Y_pred_test = model.predict(X_test)
        Y_pred_test_labels = get_labels(Y_pred_test)
        Y_pred_train = model.predict(X_train)
        Y_pred_train_labels = get_labels(Y_pred_train)

        # in einstein summation notation, take dot product of Y and Y_pred_labels to get number of correct predictions
        N_correct_test = np.sum(np.einsum('ij,ij->i', Y_test, Y_pred_test_labels))
        N_correct_train = np.sum(np.einsum('ij,ij->i', Y_train, Y_pred_train_labels))

        print(N_correct_test / len(Y_pred_test))
        print(N_correct_train / len(Y_pred_train))

        return [[N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)], [N_correct_test, N_correct_train], [len(Y_pred_test), len(Y_pred_train)]]

    else: # checking entirely new dataset
        print(prepare_data(datapath, file, input_method, task, split, p))
        X, Y = prepare_data(datapath, file, input_method, task, split, p)
        Y_pred = model.predict(X)
        Y_pred_labels = get_labels(Y_pred)
        
        # take dot product of Y and Y_pred_labels to get number of correct predictions
        N_correct = np.sum(np.einsum('ij,ij->i', Y, Y_pred_labels))
        print(N_correct / len(Y_pred))
        return [[N_correct / len(Y_pred)], [N_correct], [len(Y_pred)]]
      
            
if __name__ == '__main__':
    from xgboost import XGBRegressor
    import keras

    ## load models ##
    SH0_PATH = 'random_gen/models/stokes_h0'
    SH2_PATH = 'random_gen/models/stokes_h2'
    PH0_PATH = 'random_gen/models/prob_h0'
    PH2_PATH = 'random_gen/models/prob_h2'
    MPATHS = [SH0_PATH, SH2_PATH, PH0_PATH, PH2_PATH]
    DATA_PATH = 'random_gen/data'

    model_names = ['XGB', 'NN1', 'NN3', 'NN5']

    def load_perf():
       
        mtl = int(input('Enter 0 for s_h0, 1 for s_h2, 2 for p_h0, 3 for p_h2: '))
        m = ['s_h0', 's_h2', 'p_h0', 'p_h2']
        ew = input('Enter w for witness, e for entangled: ')

        assert mtl in [0,1, 2, 3], f'Invalid input. Must be 0, 1, 2, or 3. You entered {mtl}'
        assert ew in ['w', 'e'], f'Invalid input. Must be w or e. You entered {ew}'

        if mtl < 2: # stokes

            xgb = XGBRegressor()
            xgb.load_model(join(MPATHS[mtl], 'xgb_'+ew+'_'+m[mtl]+'.json'))
            nn1 = keras.models.load_model(join(MPATHS[mtl], 'nn1_'+ew+'_'+m[mtl]+'.h5'))
            nn3 = keras.models.load_model(join(MPATHS[mtl], 'nn3_'+ew+'_'+m[mtl]+'.h5'))
            nn5 = keras.models.load_model(join(MPATHS[mtl], 'nn5_'+ew+'_'+m[mtl]+'.h5'))
            models = [xgb, nn1, nn3, nn5]

            im = int(input(f'You have loaded stokes model {m[mtl]}. What input_method? Enter 0 for stokes_diag: '))
            assert im in [0], f'Invalid input. Must be 0. You entered {im}'

            if m[mtl]=='s_h2':
                file_split = [('hurwitz_all_4400000_b0_method_2.csv', 1), ('hurwitz_all_4400000_b0_method_0.csv', 0), ('hurwitz_all_4400000_b0_method_1.csv', 0)]
            elif m[mtl]=='s_h0':
                file_split = [('hurwitz_all_4400000_b0_method_0.csv', 1), ('hurwitz_all_4400000_b0_method_1.csv', 0), ('hurwitz_all_4400000_b0_method_2.csv', 0)]
            
            input_method = 'stokes_%i'%im

        else: # prob
            im = int(input(f'You have loaded prob model {m[mtl]}. What input_method? Enter the num of probabilities: '))
            assert im in [3, 5, 6, 9, 12, 15], f'Invalid input. You entered {im}'


            xgb = XGBRegressor()
            xgb.load_model(join(MPATHS[mtl], 'xgb_'+ew+'_prob_'+str(im)+'_'+m[mtl].split('_')[1]+'.json'))
            nn1 = keras.models.load_model(join(MPATHS[mtl], 'nn1_'+ew+'_prob_'+str(im)+'_'+m[mtl].split('_')[1]+'.h5'))
            nn3 = keras.models.load_model(join(MPATHS[mtl], 'nn3_'+ew+'_prob_'+str(im)+'_'+m[mtl].split('_')[1]+'.h5'))
            nn5 = keras.models.load_model(join(MPATHS[mtl], 'nn5_'+ew+'_prob_'+str(im)+'_'+m[mtl].split('_')[1]+'.h5'))
            models = [xgb, nn1, nn3, nn5]

            if m[mtl]=='p_h2':
                file_split = [('hurwitz_True_4400000_b0_method_2.csv', 1), ('hurwitz_True_4400000_b0_method_0.csv', 0), ('hurwitz_True_4400000_b0_method_1.csv', 0)]
            elif m[mtl]=='p_h0':
                file_split = [('hurwitz_True_4400000_b0_method_0.csv', 1), ('hurwitz_True_4400000_b0_method_1.csv', 0), ('hurwitz_True_4400000_b0_method_2.csv', 0)]

            input_method = 'prob_%i'%im

        # initialize df
        acc_df = pd.DataFrame({'model':[], 'data':[], 'acc_test':[], 'acc_train':[], 'acc_all':[], 'num_test': [], 'num_train': [], 'num_all': [], 'N_correct_test':[], 'N_correct_train':[], 'N_correct_all':[]})
        for fs in file_split:
            for i, model in enumerate(models):
                # try:
                print('model', model_names[i], 'file', fs[0], 'split', fs[1])
                acc_ls = evaluate_perf(model=model, split=bool(fs[1]), datapath=DATA_PATH, file=fs[0], input_method=input_method, task=ew)
                if len(acc_ls[0])>1:
                    acc_df = pd.concat([acc_df, pd.DataFrame.from_records([{'model':model_names[i]+m[mtl], 'data':'h'+fs[0].split('_')[-1][0], 'acc_test':acc_ls[0][0], 'acc_train':acc_ls[0][1], 'acc_all':[], 'num_test': acc_ls[2][0], 'num_train': acc_ls[2][1], 'num_all': [], 'N_correct_test':acc_ls[1][0], 'N_correct_train':acc_ls[1][1], 'N_correct_all':[]}])])
                else:
                    acc_df = pd.concat([acc_df, pd.DataFrame.from_records([{'model':model_names[i]+m[mtl], 'data':'h'+fs[0].split('_')[-1][0], 'acc_test':[], 'acc_train':[], 'acc_all':acc_ls[0][0], 'num_test': [], 'num_train': [], 'num_all': acc_ls[2][0], 'N_correct_test':[], 'N_correct_train':[], 'N_correct_all':acc_ls[1][0]}])])
                
                # except:
                #     print('Failed to evaluate model %s on file %s'%(str(model), fs[0]))
                #     continue

                
                
                

        acc_df.to_csv(join(MPATHS[mtl], 'acc_'+ew+'_'+m[mtl]+'.csv'), index=False)
        print('Saved to %s'%join(MPATHS[mtl], 'acc_'+ew+'_'+m[mtl]+'.csv'))



    load_perf()

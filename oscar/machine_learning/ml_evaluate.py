# file to evaluate performance of machine learning models
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from train_prep import prepare_data

def evaluate_perf(model,split, datapath, file, savename, input_method, task, threshold=0.4, p=0.8):
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
    if split:
        X_train, Y_train, X_test, Y_test = prepare_data(datapath, file, savename, input_method, task, split, p)
        Y_pred_test = model.predict(X_test)
        Y_pred_train = model.predict(X_train)

        # make confusion matrix
        if task=='w':
            cm_test = multilabel_confusion_matrix(Y_test, Y_pred_test)
            cm_train = multilabel_confusion_matrix(Y_train, Y_pred_train)
            # cm_test = []
            # cm_train = []
            class_labels = ['Wp_t1', 'Wp_t2', 'Wp_t3']
        elif task=='e':
            cm_test = confusion_matrix(Y_test, Y_pred_test)
            cm_train = confusion_matrix(Y_train, Y_pred_train)
            # cm_test = []
            # cm_train =[]
            class_labels = ['entangled', 'separable']
        
        # fig, axes = plt.subplots(1,2, figsize=(12,6))
        # sns.heatmap(cm_test, annot=True, ax=axes[0], fmt='g', xticklabels=class_labels, yticklabels=class_labels)
        # axes[0].set_title('Test Data')
        # sns.heatmap(cm_train, annot=True, ax=axes[1], fmt='g', xticklabels=class_labels, yticklabels=class_labels)
        # axes[1].set_title('Train Data')
        # plt.suptitle(f'Confusion Matrices for {savename}')

        # plt.tight_layout()
        # plt.savefig(join(datapath, 'confusion_matrix.pdf'))
        # plt.show()

        N_correct_test = 0
        N_correct_train = 0
        for i, y_pred in enumerate(Y_pred_test):
            if Y_test[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
                N_correct_test+=1
        
        for i, y_pred in enumerate(Y_pred_train):
            if Y_train[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
                N_correct_train+=1
        return [N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)], [cm_test, cm_train], class_labels

    else:
        X, Y = prepare_data(datapath, file, savename, input_method, task, split, p)
        Y_pred = model.predict(X)
        # Y_pred_labels = Y_pred > threshold
        # Y_pred_labels = Y_pred_labels.astype(int)
        # print(Y_pred_labels)
        # print('----------------')
        # print(Y)

        # make confusion matrix
        if task=='w':
            Y_pred_labels = Y_pred > .34
            Y_pred_labels = Y_pred_labels.astype(int)
            cm = multilabel_confusion_matrix(Y, Y_pred_labels)
            class_labels = ['Wp_t1', 'Wp_t2', 'Wp_t3']
        elif task=='e':
            # print(Y)
            # Y_sum = Y.sum(axis=1)
            # print(np.min(Y_sum))
            # print(np.max(Y_sum))
            Y_argmax = np.argmax(Y, axis=1)
            print('--------')
            Y_pred_argmax = np.argmax(Y_pred, axis=1)
            # print(argmax_indices)
            # Y_pred_labels = np.zeros(Y_pred.shape)
            # Y_pred_labels[np.arange(Y_pred.shape[0]), argmax_indices] = 1
            # Y_pred_labels = Y_pred_labels.astype(int)
            # Y_pred_sum = Y_pred_labels.sum(axis=1)
            # print(np.min(Y_pred_sum))
            cm = confusion_matrix(Y_argmax, Y_pred_argmax)
            class_labels = ['entangled', 'separable']
            # cm= []
        
        # plot confusion matrix
        # fig, ax = plt.subplots(figsize=(6,6))
        # sns.heatmap(cm, annot=True, ax=ax, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
        # ax.set_title(f'Confusion Matrix for {savename}')

        N_correct = 0
        for i, y_pred in enumerate(Y_pred):
            if Y[i][np.argmax(y_pred)]==1:
                N_correct+=1
        return N_correct / len(Y_pred), cm, class_labels
                
    # if method == 'witness':
    #     Ud = Y_test.sum(axis=1) # undetectables: count the number of states w negative witness value
    #     return [N_correct_test / (len(Y_pred_test) - len(Ud[Ud==0])), N_correct_train / (len(Y_pred_train) - len(Ud[Ud==0]))]
    # elif method == 'entangled':
    #     return [N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)]

def evaluate_perf_multiple(models,title_name, split, datapath, file, savename, input_method, task, names= ['XGB', 'NN3', 'NN5'], p=0.8):
    ''' Function to build table of 3d confusion matrices for multiple models.
    Params:
        models: list of models already properly loaded
        title_name: string for overall title
        split: boolean for whether to split data into train and test
        datapath: path to csv
        savename: what to name the prepped data
        input_method: how to prepare the input data: e.g., diag for XX, YY, ZZ
        task: 'witness' or 'entangled' for prediction method
        names: list of names for each model
        p: fraction of data to use for training
        '''
    if not(split):
        acc_df = pd.DataFrame(columns=['Model', 'Accuracy'])
        # fig, axes = plt.subplots(len(models), 1, figsize=(12,6), subplot_kw={'projection': '3d'})
        fig, axes = plt.subplots(len(models), 1, figsize=(6,12))
        for i, model in enumerate(models):
            acc, cm, class_labels = evaluate_perf(model, split, datapath, file, savename, input_method, task, p)
            acc_df = pd.concat([acc_df, pd.DataFrame.from_records([{'Model': names[i], 'Accuracy':acc}])])
            print(names[i], acc)
            # print(cm.shape)
            # print(cm)
            cm_norm =cm /  np.sum(cm) #normalize confusion matrix
            # plot confusion matrix--map predicted and actual targs to their height and color 
            # print(cm_norm)
            # z, c = [], []
            # for j in range(len(class_labels)):
            #     for k in range(len(class_labels)):
            #         z.append(cm[j, k])
            #         c.append(cm_norm[j,k])
            # print('z', z)
            # print('c', c)
            # z = np.array(z).resize((len(class_labels), len(class_labels)))
            # dz = np.zeros((len(class_labels), len(class_labels)))
        
            # b3d = axes[i].bar3d(x, y, np.zeros_like(cm), 1, 1, cm, shade=True, color=cm_norm.flatten(), alpha=0.8)
            # axes[i].set_title(f'{names[i]}: {acc}')
            # axes[i].set_xticks(np.arange(0, len(class_labels), 1))
            # axes[i].set_xticklabels(labels + '_true' for labels in class_labels)
            # axes[i].set_yticks(np.arange(0, len(class_labels), 1))
            # axes[i].set_yticklabels(labels + '_pred' for labels in class_labels)
            # axes[i].set_zlabel('Count')
            # cb = fig.colorbar(b3d, plt.cm.ScalarMappable(norm=None, cmap='viridis'), vmin = np.min(c), vmax=np.max(c), ax=axes[i], shrink=0.6)
            # cb.set_label('Normalized Count')
            # cb.ax.set_position(cb.ax.get_position().translated(0.09, 0))
            sns.heatmap(cm_norm, annot=True, ax=axes[i], fmt='g', xticklabels=class_labels, yticklabels=class_labels)
            axes[i].xaxis.set_ticklabels(class_labels)
            axes[i].yaxis.set_ticklabels(class_labels)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_title('%s: %.3g' % (names[i], acc))
        plt.suptitle(f'Confusion matrices for {title_name}')
        plt.tight_layout()
        plt.savefig(join(datapath, f'confusion_matrix_{savename}.pdf'))
        plt.show()
        acc_df.to_csv(join(datapath, f'accuracy_{savename}.csv'))
    else:
        acc_df = pd.DataFrame(columns=['Model', 'Test Accuracy', 'Train Accuracy'])
        for i, model in enumerate(models):
            acc, cm, class_labels = evaluate_perf(model, split, datapath, file, savename, input_method, task, p)
            acc_df = pd.concat([acc_df, pd.DataFrame.from_records([{'Model': names[i], 'Test Accuracy':acc[0], 'Train Accuracy':acc[1]}])])
        acc_df.to_csv(join(datapath, f'accuracy_{title_name}.csv'))  
            
if __name__ == '__main__':
    from xgboost import XGBRegressor
    import keras

    ## load models ##
    SH0_PATH = 'random_gen/models/stokes_h0'
    SH2_PATH = 'random_gen/models/stokes_h2'
    MPATHS = [SH0_PATH, SH2_PATH]
    DATA_PATH = 'random_gen/data'

    def load_perf(debug=False):
        if not(debug):
            mtl = int(input('Enter 0 for s_h0, 1 for s_h2: '))
            m = ['s_h0', 's_h2']
            ew = input('Enter w for witness, e for entangled: ')

            assert mtl in [0,1], f'Invalid input. Must be 0 or 1. You entered {mtl}'
            assert ew in ['w', 'e'], f'Invalid input. Must be w or e. You entered {ew}'

            xgb = XGBRegressor()
            xgb.load_model(join(MPATHS[mtl], 'xgb_'+ew+'_'+m[mtl]+'.json'))
            nn3 = keras.models.load_model(join(MPATHS[mtl], 'nn3_'+ew+'_'+m[mtl]+'.h5'))
            nn5 = keras.models.load_model(join(MPATHS[mtl], 'nn5_'+ew+'_'+m[mtl]+'.h5'))
            models = [xgb, nn3, nn5]

            if mtl < 2: # stokes
                im = int(input(f'You have loaded stokes model {m[mtl]}. What input_method? Enter 0 for stokes_diag: '))
                ftl = int(input('Load test data, enter 0 for method 0, 1 for method 1, 2 for method 2: '))

                assert im in [0], f'Invalid input. Must be 0. You entered {im}'
                assert ftl in [0,1,2], f'Invalid input. Must be 0, 1, or 2. You entered {ftl}'

                file = 'hurwitz_all_4400000_b0_method_%i.csv'%ftl
                input_method = 'stokes_%i'%im
                split = bool(int(input('Enter 1 to split data, 0 to not split: ')))

            else: # prob
                im = int(input('You have loaded stokes model {m[mtl]}. What input_method? Enter the num of probabilities: '))
                ftl = int(input(f'You have loaded prob model {m[mtl]}. To load test data, enter 0 for method 0, 1 for method 1, 2 for method 2: '))

                assert im in [3, 5, 6, 9, 12, 15], f'Invalid input. You entered {im}'
                assert ftl in [0,1,2], f'Invalid input. Must be 0, 1, or 2. You entered {ftl}'

                file = 'hurwitz_True_all_4400000_b0_method_%i.csv'%ftl
                input_method = 'prob_%i'%im
                split = bool(int(input('Enter 1 to split data, 0 to not split: ')))

            title_name = f'Model Trained on {m[mtl]}, Testing on {ftl}, {ew}'
            save_name = f'{m[mtl]}_{ftl}_{ew}'
        else: # debug mode: s_h2 loaded on method 0
            ew = 'w'
            xgb= XGBRegressor()
            xgb.load_model(join(MPATHS[1], 'xgb_'+ew+'_s_h2.json'))
            nn3 = keras.models.load_model(join(MPATHS[1], 'nn3_'+ew+'_s_h2.h5'))
            nn5 = keras.models.load_model(join(MPATHS[1], 'nn5_'+ew+'_s_h2.h5'))
            models = [xgb, nn3, nn5]
            im = 0
            ftl = 0 # test on method 0
            file = 'hurwitz_all_4400000_b0_method_%i.csv'%ftl
            input_method = 'stokes_%i'%im
            split = False
            title_name = f'Trained on s_h2, Testing on {ftl}, {ew}'
            save_name = f's_h2_{ftl}_{ew}'

        evaluate_perf_multiple(models=models, title_name=title_name, split=split, datapath=DATA_PATH, file=file, savename=save_name, input_method=input_method, task=ew)

if __name__ == '__main__':
    load_perf(True)

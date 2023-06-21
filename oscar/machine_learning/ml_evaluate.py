# file to evaluate performance of machine learning models
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt

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
        if task=='witness':
            # cm_test = multilabel_confusion_matrix(Y_test, Y_pred_test)
            # cm_train = multilabel_confusion_matrix(Y_train, Y_pred_train)
            cm_test = []
            cm_train = []
            class_labels = ['Wp_t1', 'Wp_t2', 'Wp_t3']
        elif task=='entangled':
            # cm_test = confusion_matrix(Y_test, Y_pred_test)
            # cm_train = confusion_matrix(Y_train, Y_pred_train)
            cm_test = []
            cm_train =[]
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
        Y_pred_labels = Y_pred > threshold
        Y_pred_labels = Y_pred_labels.astype(int)
        print(Y_pred_labels)
        print('----------------')
        print(Y)

        # make confusion matrix
        if task=='witness':
            # cm = multilabel_confusion_matrix(Y, Y_pred_labels)
            cm =[]
            class_labels = ['Wp_t1', 'Wp_t2', 'Wp_t3']
        elif task=='entangled':
            # cm = confusion_matrix(Y, Y_pred_labels)
            class_labels = ['entangled', 'separable']
            cm= []
        
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

def evaluate_perf_multiple(models,names, title_name, split, datapath, file, savename, input_method, task, p=0.8):
    ''' Function to build table of 3d confusion matrices for multiple models.
    Params:
        models: list of models already properly loaded
        names: list of names for each model
        title_name: string for overall title
        split: boolean for whether to split data into train and test
        datapath: path to csv
        savename: what to name the prepped data
        input_method: how to prepare the input data: e.g., diag for XX, YY, ZZ
        task: 'witness' or 'entangled' for prediction method
        p: fraction of data to use for training
        '''
    if not(split):
        acc_df = pd.DataFrame(columns=['Model', 'Accuracy'])
        fig, axes = plt.subplots(len(models), 1, figsize=(12,6))
        for i, model in enumerate(models):
            acc, cm, class_labels = evaluate_perf(model, split, datapath, file, savename, input_method, task, p)
            acc_df = pd.concat([acc_df, pd.DataFrame.from_records([{'Model': names[i], 'Accuracy':acc}])])
        #     print(names[i], acc)
        #     print(cm)
        #     cm_norm =cm /  np.linalg.norm(cm, axis=0) # normalize confusion matrix
        #     # plot confusion matrix--map predicted and actual targs to their height and color 
        #     z, c = [], []
        #     for j in range(len(class_labels)):
        #         for k in range(len(class_labels)):
        #             z.append(cm[j, k])
        #             c.append(cm_norm[j,k])
        #     z = np.array(z).shape((len(class_labels), len(class_labels)))
        #     c = np.array(c)

        #     b3d = axes[i].bar3d(np.arange(0, len(class_labels), 1), np.arange(0, len(class_labels), 1), np.zeros(len(class_labels)), 1, 1, z, shade=True, color=c)
        #     axes[i].set_title(f'{names[i]}: {acc}')
        #     axes[i].set_xticks(np.arange(0, len(class_labels), 1))
        #     axes[i].set_xticklabels(labels + '_true' for labels in class_labels)
        #     axes[i].set_yticks(np.arange(0, len(class_labels), 1))
        #     axes[i].set_yticklabels(labels + '_pred' for labels in class_labels)
        #     axes[i].set_zlabel('Count')
        #     cb = fig.colorbar(b3d, plt.cm.ScalarMappable(norm=None, cmap='viridis'), vmin = np.min(c), vmax=np.max(c), ax=axes[i], shrink=0.6)
        #     cb.set_label('Normalized Count')
        #     cb.ax.set_position(cb.ax.get_position().translated(0.09, 0))
        # plt.suptitle('Confusion matrices for ', title_name)
        # plt.tight_layout()
        # # plt.savefig(join(datapath, f'confusion_matrix_{title_name}.pdf'))
        # plt.show()
        acc_df.to_csv(join(datapath, f'accuracy_{title_name}.csv'))
    else:
        acc_df = pd.DataFrame(columns=['Model', 'Test Accuracy', 'Train Accuracy'])
        for i, model in enumerate(models):
            acc, cm, class_labels = evaluate_perf(model, split, datapath, file, savename, input_method, task, p)
            acc_df = pd.concat([acc_df, pd.DataFrame.from_records([{'Model': names[i], 'Test Accuracy':acc[0], 'Train Accuracy':acc[1]}])])
        acc_df.to_csv(join(datapath, f'accuracy_{title_name}.csv'))  
            
if __name__ == '__main__':
    from xgboost import XGBRegressor
    import keras
    MODEL_PATH = 'random_gen/models/6_18'

    s_h2_xgb = XGBRegressor()
    s_h2_xgb.load_model(join(MODEL_PATH, 'xgb_e_h2_0.json'))
    s_h2_3nn = keras.models.load_model(join(MODEL_PATH, 'nn3_e_h2_0.h5'))
    s_h2_5nn = keras.models.load_model(join(MODEL_PATH, 'nn5_e_h2_0.h5'))
    evaluate_perf_multiple(models=[s_h2_xgb, s_h2_3nn, s_h2_5nn],names=['XGB', 'NN3', 'NN5'], title_name='Entangled, Train on Stokes Hurwitz Method 2, Evaluating Method 0', split=True, datapath='random_gen/data', file='hurwitz_all_4400000_b0_method_0.csv', savename='test', input_method='stokes_diag', task='entangled')

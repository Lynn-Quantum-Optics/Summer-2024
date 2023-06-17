# file to evaluate performance of machine learning models
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from train_prep import prepare_data

def evaluate_perf(model,split, datapath, file, savename, input_method, task, p=0.8):
    ''' Function to measure accuracy on both train and test data.
    Params:
        model: trained model
        split: boolean for whether to split data into train and test
        datapath: path to csv
        savename: what to name the prepped data
        input_method: how to prepare the input data: e.g., diag for XX, YY, ZZ
        task: 'witness' or 'entangled' for prediction method
        p: fraction of data to use for training
        '''
    if split:
        X_train, Y_train, X_test, Y_test = prepare_data(datapath, file, savename, input_method, task, split, p)
        Y_pred_test = model.predict(X_test)
        Y_pred_train = model.predict(X_train)

        # make confusion matrix
        if task=='witness':
            cm_test = multilabel_confusion_matrix(Y_test, Y_pred_test)
            cm_train = multilabel_confusion_matrix(Y_train, Y_pred_train)
            class_labels = ['Wp_t1', 'Wp_t2', 'Wp_t3']
        elif task=='entangled':
            cm_test = confusion_matrix(Y_test, Y_pred_test)
            cm_train = confusion_matrix(Y_train, Y_pred_train)
            class_labels = ['entangled', 'separable']
        
        fig, axes = plt.subplots(1,2, figsize=(12,6))
        sns.heatmap(cm_test, annot=True, ax=axes[0], fmt='g', xticklabels=class_labels, yticklabels=class_labels)
        axes[0].set_title('Test Data')
        sns.heatmap(cm_train, annot=True, ax=axes[1], fmt='g', xticklabels=class_labels, yticklabels=class_labels)
        axes[1].set_title('Train Data')
        plt.suptitle(f'Confusion Matrices for {savename}')

        plt.tight_layout()
        plt.savefig(join(datapath, 'confusion_matrix.pdf'))
        plt.show()

        N_correct_test = 0
        N_correct_train = 0
        for i, y_pred in enumerate(Y_pred_test):
            if Y_test[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
                N_correct_test+=1
        
        for i, y_pred in enumerate(Y_pred_train):
            if Y_train[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
                N_correct_train+=1
        return [N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)]

    else:
        X, Y = prepare_data(datapath, file, savename, input_method, task, split, p)
        Y_pred = model.predict(X)

        # make confusion matrix
        if task=='witness':
            cm = multilabel_confusion_matrix(Y, Y_pred)
            class_labels = ['Wp_t1', 'Wp_t2', 'Wp_t3']
        elif task=='entangled':
            cm = confusion_matrix(Y, Y_pred)
            class_labels = ['entangled', 'separable']
        
        # plot confusion matrix
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
        ax.set_title(f'Confusion Matrix for {savename}')

        N_correct = 0
        for i, y_pred in enumerate(Y_pred):
            if Y[i][np.argmax(y_pred)]==1:
                N_correct+=1
        return N_correct / len(Y_pred)
                
    # if method == 'witness':
    #     Ud = Y_test.sum(axis=1) # undetectables: count the number of states w negative witness value
    #     return [N_correct_test / (len(Y_pred_test) - len(Ud[Ud==0])), N_correct_train / (len(Y_pred_train) - len(Ud[Ud==0]))]
    # elif method == 'entangled':
    #     return [N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)]



# mlp for multi-output regression
import numpy as np
from numpy import mean, loadtxt, std, argmax
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import tensorflow as tf
import keras_tuner

"""
This code is heavily based on Jason Brownless's tutorial linked below
https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/#:~:text=Multi%2Doutput%20regression%20is%20a,a%20prediction%20for%20new%20data.

Tutorial to create custom loss function: 
https://towardsdatascience.com/how-to-create-a-custom-loss-function-keras-3a89156ec69b 
https://cnvrg.io/keras-custom-loss-functions/

https://stackoverflow.com/questions/35146444/tensorflow-python-accessing-individual-elements-in-a-tensor 
https://www.tensorflow.org/guide/tensor
https://towardsdatascience.com/how-to-replace-values-by-index-in-a-tensor-with-tensorflow-2-0-510994fe6c5f 
https://neptune.ai/blog/keras-loss-functions
https://www.kdnuggets.com/2019/04/advanced-keras-constructing-complex-custom-losses-metrics.html
https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/ 
https://blog.manash.io/multi-task-learning-in-keras-implementation-of-multi-task-classification-loss-f1d42da5c3f6 

"""

"""
using same architecture of CNN as Roik
"""
class CustomMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="custom_metric", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype=tf.int32, initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = witness_loss_fn(y_pred, y_true)
        count = tf.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values *= sample_weight
            count *= sample_weight
        self.sum.assign_add(values)
        self.count.assign_add(count)

    def result(self):
        return self.sum / tf.cast(self.count, tf.float32)

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)
    
def witness_loss_fn(y_true, y_pred):
    # y_pred is a 1 by batch size tensor with all the predicted values for each state
    # y_true is a 3 by batch size tensor, with the three ground truth outputs per state

    batchsize = y_true.shape[0]
    
    
    # making y_pred 1 output when evaluating loss
    y_pred_xy = y_pred[0]
    # print('y_pred_xy', str(y_pred_xy))
    y_pred_yz = y_pred[1]
    # print('y_pred_yz', str(y_pred_yz))
    y_pred_xz = y_pred[2]
    # print('y_pred_xz', str(y_pred_xz))

    # for y_true, getting labels
    xy_qual = y_true[0]
    yz_qual = y_true[1]
    xz_qual = y_true[2]

    loss = y_pred_xy*xy_qual + y_pred_yz*yz_qual + y_pred_xz*xz_qual

    return tf.reduce_mean(loss, axis=-1)
   


# # split the dataset into a training set and a validation set
# train_df, val_df = train_test_split(df, test_size=0.05)

def get_dataset(dataset):
    # split into input (X) and output (y) variables
    # X = dataset.loc[:,'HH probability':'LL probability']
    # y = dataset.loc[:,'XY min': 'XZ min']
    X = dataset.loc[:, 'HH': 'LL']
    y = dataset.loc[:, 'Wp_t1': 'Wp_t3']
    return X, y

# get the model, in this case there n_inputs = 12, n_outputs = 3
# def get_model(n_inputs, n_outputs):
#     model = Sequential()
#     model.add(Dense(100, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
#     model.add(Dense(40, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
#     model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
#     model.add(Dense(10, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
#     model.add(Dense(n_outputs, activation = 'softmax'))
#     opt = keras.optimizers.Adam(clipnorm=1.0)
#     model.compile(loss='witness_loss_fn', optimizer=opt)
#     return model

def get_model(hp):
    model = Sequential()
    model.add(
        keras.layers.Dense(
            units = hp.Int("units", min_value = 30, max_value = 400, step= 50),
            activation = 'relu',
            )
        )
    model.add(
        keras.layers.Dense(
            units = hp.Int("units", min_value = 30, max_value = 400, step= 50),
            activation = 'relu',
            )
        )
    model.add(
        keras.layers.Dense(
            units = hp.Int("units", min_value = 50, max_value = 400, step= 50),
            activation = 'relu',
            )
        )
    model.add(
        keras.layers.Dense(
            units = hp.Int("units", min_value = 50, max_value = 400, step= 50),
            activation = 'relu',
            )
        )
    model.add(Dense(3, activation = 'softmax'))
    opt = keras.optimizers.Adam(clipnorm=1.0)
    model.compile(
        loss=witness_loss_fn, optimizer=opt, metrics = [CustomMetric()])
    return model

"""
Using k-fold cross-validation is especially useful for smaller datasets, 
this is a pretty big dataset but this technique can still be pretty useful
"""
# evaluate a model using repeated k-fold cross-validation
# def evaluate_model(X, y):
#     results = list()
#     n_inputs, n_outputs = X.shape[1], y.shape[1]
#     # define evaluation procedure
#     cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#     # enumerate folds

#     for train_idx, test_idx in cv.split(X):
#         # prepare data
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#         # define model
#         model = get_model(n_inputs, n_outputs, keras_tuner.HyperParameters())
#         # fit model
#         model.fit(X_train, y_train, verbose=0, epochs=100)
#         # evaluate model on test set
#         mae = model.evaluate(X_test, y_test)
#         # store result
#         print('>%.3f' % mae)
#         results.append(mae)
#     return results


# """
# The data files contains 10,000 randomly generated states 
# with the values for W1-W6 min and max and whether or not 
# it is detected by one of the triplet prime witnesses
# """

# data_1 = 'all_qual_2000.csv'
# data_2 = 'S22_data/all_qual_20000_1.csv'
# data_3 = 'all_qual_20000_2.csv'
# data_4 = 'all_qual_20000_3.csv'
# data_5 = 'all_qual_20000_4.csv'
# data_6 = 'all_qual_20000_5.csv'

data_method0 = 'random_gen/data/hurwitz_True_4400000_b0_method_0.csv'
df= pd.read_csv(data_method0)
df = df.loc[(df['W_min']>=0) & ((df['Wp_t1']<0) | (df['Wp_t2']<0) | (df['Wp_t3']<0))]

# df_1 = pd.read_csv(data_1)
# df = pd.read_csv(data_2)
# df_3 = pd.read_csv(data_3)
# df_4 = pd.read_csv(data_4)
# df_5 = pd.read_csv(data_5)
# df_6 = pd.read_csv(data_6)

# df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6])

#load dataset
X, y = get_dataset(df)


tuner = keras_tuner.RandomSearch(
    hypermodel=get_model,
    objective = keras_tuner.Objective("val_custom_metric", direction = "min"),
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    # directory="my_dir",
    # project_name="helloworld",
)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
tuner.search(X_train, y_train, epochs=5, validation_data = (X_val, y_val))

models = tuner.get_best_models(num_models=5)
best_hps = tuner.get_best_hyperparameters(20)
model = get_model(best_hps[0])

model.fit(X,y, epochs=100)



# # # Compile model
# opt = keras.optimizers.Adam(clipnorm=1.0)
# model = get_model(12,3, keras_tuner.HyperParameters())
# model.compile(loss=witness_loss_fn, optimizer=opt, metrics=[witness_loss_fn], direction = "min")


# Fit the model
# model.fit(X, y, epochs=150, batch_size=50)

# # evaluate the model
# scores = model.evaluate(X, y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], 100*scores[1]))


# # # serialize model to JSON
model_json = model.to_json()
with open("model_qual_v3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_qual_v3.h5")
print("Saved model to disk")
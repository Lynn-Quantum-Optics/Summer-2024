# file to revamp becca and laney's neural net from last semester
import os
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import sem
import numpy as np
import pandas as pd

# read in and prepare data
DATA_PATH = 'S22_data'

def prepare_data(df_path_ls, random_seed): # do target prep is binary
    print('preparing data.....')
    df_ls = []
    for df_path in df_path_ls:
        df_ls.append(pd.read_csv(join(DATA_PATH, df_path)))
    
    df = pd.concat(df_i for df_i in df_ls)
    df = df.sample(frac=1, random_state=random_seed).reset_index() # randomize rows

    # define inputs and outputs for model
    inputs = ['HH probability', 'HV probability', 'VH probability', 'VV probability',
        'DD probability', 'DA probability', 'AD probability', 'AA probability',
        'RR probability', 'RL probability', 'LR probability', 'LL probability']
    outputs = ['XY min', 'YZ min','XZ min']

    # function to simplify the outputs by replacing the most negative value with 1 and setting others to 0
    def simplify_targ(df):
        df_t = df.T # transpose original to get argmin on each column
        # df_out = pd.DataFrame(columns=[i for i in range(len(df_t))])

        def get_out_vec(values):
            # set sensitivity for how close 2 witness values need to be
            
            # targ = values.argmin() # assign value based on minimum
            # out_vec = np.zeros(len(outputs))
            # out_vec[targ]=1

            out_vec = np.zeros(len(values))
            out_vec[values < 0] = 1 # get all negatives
            return out_vec

        # for _, values in df_t.iteritems():
        #     df_out.loc[len(df_out)] = get_out_vec(values)
        #     # print(df_out)
        # df_out = df_t[df_t]

        df_out = df.applymap(lambda x: 1 if x < 0 else 0)

        return df_out

    # function to build the dataset: p is proportion in training vs test; inputs is list of strings of input states, same for outputs
    def prep_data_targ(df, p, inputs, outputs):
        split_index = int(p*len(df))
        df_train = df.iloc[:split_index, :]
        df_test = df.iloc[split_index:, :]

        X_train = df_train[inputs]
        Y_train = simplify_targ(df_train[outputs])
        # Y_train = df_train[outputs]

        X_test = df_test[inputs]
        Y_test = simplify_targ(df_test[outputs])

        return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()

    # functions to build and train the network
    split_frac = 0.8 # split some portion to be train vs test
   
    X_train, Y_train, X_test, Y_test = prep_data_targ(df, split_frac, inputs, outputs)

    # save!
    np.save(join(DATA_PATH, 'x_train'), X_train)
    np.save(join(DATA_PATH,'y_train'), Y_train)
    np.save(join(DATA_PATH,'x_test'), X_test)
    np.save(join(DATA_PATH,'y_test'), Y_test)


# read in data
def read_data():
    X_train = np.load(join(DATA_PATH, 'x_train.npy'), allow_pickle=True)
    Y_train = np.load(join(DATA_PATH, 'y_train.npy'), allow_pickle=True)
    X_test = np.load(join(DATA_PATH, 'x_test.npy'), allow_pickle=True)
    Y_test = np.load(join(DATA_PATH, 'y_test.npy'), allow_pickle=True)

    return X_train, Y_train, X_test, Y_test

## call to load and save dfs ##
df_path_ls = []
for path in os.listdir(DATA_PATH):
    if path.endswith('.csv'):
        df_path_ls.append(path)

## code for neural network ##
def do_nn(X_train, Y_train, X_test, Y_test):

    import tensorflow as tf
    from keras import layers
    from keras.models import Sequential
    from keras.optimizers import Adam

    ## model creation ##
    output_len = 3
    def build_model_test(size=50, dropout=0.1, learning_rate=0.001):

        def witness_loss_fn(y_true, y_pred):
            # y_pred is a 1 by batch size tensor with all the predicted values for each state
            # y_true is a 3 by batch size tensor, with the three ground truth outputs per state
            
            # making y_pred 1 output when evaluating loss
            y_pred_xy = y_pred[0]
            y_pred_yz = y_pred[1]
            y_pred_xz = y_pred[2]

            # for y_true, getting labels
            xy_qual = y_true[0]
            yz_qual = y_true[1]
            xz_qual = y_true[2]

            loss = y_pred_xy*xy_qual + y_pred_yz*yz_qual + y_pred_xz*xz_qual

            return tf.reduce_mean(loss, axis=-1)

        model = Sequential()

        model.add(layers.Dense(size))
        model.add(layers.Dense(size))
        model.add(layers.Dense(size))
        model.add(layers.Dense(size))
        model.add(layers.Dense(size))

        model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(output_len))
        model.add(layers.Activation('softmax'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model

    def train_manual(X_train, Y_train, X_test, Y_test):
        global model
        model = build_model_test()
        
        #now run training
        history = model.fit(
        X_train, Y_train,
        batch_size = 100,
        validation_data=(X_test, Y_test),
        epochs=10,
        shuffle=True
        )
        return model

    ## call training ##
    model = train_manual(X_train, Y_train, X_test, Y_test)
    return model

def do_xgboost(X_train, Y_train, X_test, Y_test):
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error

    # define the model
    model = XGBRegressor(n_estimators = 1000, random_state=0) 

    # fit the model
    model.fit(X_train, Y_train, early_stopping_rounds = 10, eval_set = [(X_test, Y_test)]) 

    # make prediction
    # predictions = model.predict(X_test)
    # mae = mean_absolute_error(predictions, Y_test)
    # print('mae',mae)
    return model

# returns accuracy as defined by becca and laney in the spring write up
def evaluate_perf(model):
    Y_pred_test = model.predict(X_test)
    Y_pred_train = model.predict(X_train)
    N_correct_test = 0
    N_correct_train = 0
    for i, y_pred in enumerate(Y_pred_test):
        # min_guess = np.argmin(y_pred)
        # min_actual = np.argmax(Y_test[i])
        # if min_guess == min_actual:
        #     N_correct+=1
        if Y_test[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
            N_correct_test+=1
    
    for i, y_pred in enumerate(Y_pred_train):
        if Y_train[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
            N_correct_train+=1
    # return N_correct / len(X_test)
    Ud = Y_test.sum(axis=1) # undetectables: count the number of states w negative witness value
    return [N_correct_test / (len(Y_pred_test) - len(Ud[Ud==0])), N_correct_train / (len(Y_pred_train) - len(Ud[Ud==0]))] # return both the test and train results

# for testing single state
# prepare_data(df_path_ls, 0)
# X_train, Y_train, X_test, Y_test = read_data()

## initialize data ##
# pick 100 random states; test performance
random_arr = np.random.randint(1,100, size=(1,100))
train_perf =[]
test_perf = []
for rand in random_arr:
    prepare_data(df_path_ls, rand)
    X_train, Y_train, X_test, Y_test = read_data()

    # stats for xgboost
    model_xgb = do_xgboost(X_train, Y_train, X_test, Y_test)
    frac_xgb = evaluate_perf(model_xgb)
    print('fraction correct of xgb for seed '+str(rand), frac_xgb[0], frac_xgb[1])
    test_perf.append(frac_xgb[0])
    train_perf.append(frac_xgb[1])

# visualize perf
test_mean = np.mean(test_perf)
test_sem = sem(test_perf)
print('mean for test:', test_mean, 'sem for test:', test_sem)
train_mean = np.mean(train_perf)
train_sem = sem(train_perf)
print('mean for train:', train_mean, 'sem for train:', train_sem)

plt.plot(random_arr, test_perf, label='test')
plt.plot(random_arr, train_perf, label='train')
plt.xlabel('Random integer', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Performance of XGB model', fontsize=16)
plt.legend()
plt.savefig('100random_xgb.pdf')
plt.show()

# print stats for neural net
# model = do_nn(X_train, Y_train, X_test, Y_test)
# frac_me = evaluate_perf(model)
# print('fraction correct of o nn', frac_me[0], frac_me[1])

## load trained bl model ##
# json_file = open('models/model_qual_v2.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# bl_model = keras.models.model_from_json(loaded_model_json)
# # load weights into new model
# bl_model.load_weights("models/model_qual_v2.h5")

# frac_bl = evaluate_perf(bl_model)
# print('fraction correct of bl model', frac_bl[0], frac_bl[1])


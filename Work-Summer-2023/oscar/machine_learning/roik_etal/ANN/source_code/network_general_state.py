import sys
import time, datetime 
import tensorflow as tf
from tensorflow import keras
import numpy as np

from data import train_data
from data import test_data
from data import train_labels
from data import test_labels

#from vypocet_general_state import train_data_15 as train_data
#from vypocet_general_state import test_data_15 as test_data
#from vypocet_general_state import train_labels as train_labels
#from vypocet_general_state import test_labels as test_labels



#from generator_general_state import train_data
#from generator_general_state igen_values
#
#model = keras.Sequential([
#    keras.layers.Dense(36, activation="relu"),
#    keras.layers.Dense(180, activation="relu"),
#    keras.layers.Dense(75, activation="relu"),
#    keras.layers.Dense(180, activation="relu"),
#    keras.layers.Dense(75, activation="relu"),
#    keras.layers.Dense(2, activation="softmax")])


model = keras.Sequential([
    keras.layers.Dense(360, activation="relu"),
    keras.layers.Dense(1800, activation="relu"),
    keras.layers.Dense(750, activation="relu"),
    keras.layers.Dense(1800, activation="relu"),
    keras.layers.Dense(750, activation="relu"),
    keras.layers.Dense(2, activation="softmax")])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_data, train_labels, epochs=60)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print("Tested Acc:", test_acc)  

model.save("model_15_final.h5")


#model = keras.models.load_model("test_model_6.h5")
#
#
#
#prediction = model.predict(train_data)
#
#counter = 0
#fin = len(train_data)
#vim_90 = 0
#vim_80 = 0
#nevim_90 = 0
#nevim_80 = 0
#chyba_90 = 0
#chyba_80 = 0
#vim_50 = 0
#with open("uvidime.txt",'w') as f :
#    while train_data:
#        h = str(round(prediction[0+counter][0],4))+"\t"+str(round(prediction[0+counter][1],4))+"\t"+str(round(igen_values[0+counter],4))
#        f.writelines(h)
#        f.write("\n")
#        if  round(prediction[0+counter][1],4)>0.9:
#            vim_90 = vim_90 + 1
#        if round(prediction[0+counter][1],4)>0.8:
#            vim_80 = vim_80 + 1
#        if   0.1 < round(prediction[0+counter][1],4) < 0.9:
#            nevim_90 = nevim_90 + 1
#        if   0.2 < round(prediction[0+counter][1],4) < 0.8:
#            nevim_80 = nevim_80 + 1
#        if   0.1 > round(prediction[0+counter][1],4):
#            chyba_90 = chyba_90 + 1
#        if   0.2 > round(prediction[0+counter][1],4):
#            chyba_80 = chyba_80 + 1
#        if   0.5 < round(prediction[0+counter][1],4):
#            vim_50 = vim_50 + 1
#        else:   
#            vim_90 = vim_90 + 0    
#        counter = counter + 1
#        if counter == (fin):
#            break
#    vim_90_txt = str(vim_90)
#    vim_80_txt = str(vim_80)
#    nevim_90_txt = str(nevim_90)
#    nevim_80_txt = str(nevim_80) 
#    chyba_90_txt = str(chyba_90)
#    chyba_80_txt = str(chyba_80)
#    vim_50_txt = str(vim_50)     
#    f.writelines(vim_90_txt)
#    f.write("\n")
#    f.writelines(vim_80_txt)
#    f.write("\n")
#    f.writelines(nevim_90_txt)
#    f.write("\n")
#    f.writelines(nevim_80_txt)
#    f.write("\n")
#    f.writelines(chyba_90_txt)
#    f.write("\n")
#    f.writelines(chyba_80_txt)
#    f.write("\n")
#    f.writelines(vim_50_txt)
#
#



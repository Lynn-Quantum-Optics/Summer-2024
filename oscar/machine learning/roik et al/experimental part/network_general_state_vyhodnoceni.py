import sys
import time, datetime 
import tensorflow as tf
from tensorflow import keras
import numpy as np

from generace_stavu import train_data
from generace_stavu import p
model = keras.models.load_model("model_5_8796.h5")

prediction = model.predict(train_data)

data_label = []
counter = 0
for predictions in prediction:
    print(p[0+counter])
    print(predictions)    
    if p[0+counter] < 0.33 and 0.90 < prediction[0+counter][1]:
        print('makam')
        data_label.append(1)     
    if p[0+counter] > 0.33 and 0.10 > prediction[0+counter][1]:
        data_label.append(0)
        print('makam')




    if p[0+counter] == 0.33 and 0.90 > prediction[0+counter][1]:
        data_label.append(0)
        print('makam')
    if p[0+counter] == 0.33 and 0.90 < prediction[0+counter][1]:
        data_label.append(1)
        print('makam')     
    counter = counter + 1


print(len(data_label))
print(len(train_data))
print(len(p))

fin = len(train_data)
vim_50 = 0
with open("statictika_final_exten.txt",'w') as f :
    counter = 0
    while counter < len(train_data):
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(data_label[0+counter])+"\t"+str(p[0+counter])
        f.writelines(h)
        f.write("\n")
        if p[0+counter] < 0.33 and 0.5 > prediction[0+counter][0]:
            vim_50 = vim_50 + 1
        if p[0+counter] > 0.33 and 0.5 > prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        counter = counter + 1
        if counter == (fin):
            break

    vim_50_txt = str(vim_50)     
    f.write("\n")
    f.writelines(vim_50_txt)




import sys
import time, datetime 
import tensorflow as tf
from tensorflow import keras
import numpy as np

#from data import train_data
#from data import test_data
#from data import train_labels
#from data import test_labels
from data_final import train_data
from data_final import train_labels
from data_final import igen_values

#model = keras.Sequential([
#    keras.layers.Dense(36, activation="relu"),
#    keras.layers.Dense(72, activation="relu"),
#    keras.layers.Dense(36, activation="relu"),
#    keras.layers.Dense(6, activation="relu"),
#    keras.layers.Dense(2, activation="softmax")])
#
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
#model.fit(train_data, train_labels, epochs=200)
#
#test_loss, test_acc = model.evaluate(test_data, test_labels)
#
#print("Tested Acc:", test_acc)  
#
#model.save("test_model_36_pokus.h5")


model = keras.models.load_model("model_12_937.h5")



prediction = model.predict(train_data)


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_1.txt",'w') as f :
    counter = 0
    while counter < 400000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 > prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)


############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_2.txt",'w') as f :
    counter = 400000
    while counter < 800000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_3.txt",'w') as f :
    counter = 800000
    while counter < 1200000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_4.txt",'w') as f :
    counter = 1200000
    while counter < 1600000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_5.txt",'w') as f :
    counter = 1600000
    while counter < 2000000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_6.txt",'w') as f :
    counter = 2000000
    while counter < 1200000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_7.txt",'w') as f :
    counter = 1200000
    while counter < 2800000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_8.txt",'w') as f :
    counter = 2800000
    while counter < 3200000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_9.txt",'w') as f :
    counter = 3200000
    while counter < 3600000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)

############################################


fin = len(train_data)
vim_90 = 0
vim_80 = 0
nevim_90 = 0
nevim_80 = 0
chyba_90 = 0
chyba_80 = 0
vim_50 = 0
with open("statictika_12_10.txt",'w') as f :
    counter = 3600000
    while counter < 4000000:
        h = str(prediction[0+counter][0])+"\t"+str(prediction[0+counter][1])+"\t"+str(igen_values[0+counter])
        f.writelines(h)
        f.write("\n")
        if  prediction[0+counter][1]>0.9:
            vim_90 = vim_90 + 1
        if prediction[0+counter][1]>0.8:
            vim_80 = vim_80 + 1
        if   0.1 < prediction[0+counter][1] < 0.9:
            nevim_90 = nevim_90 + 1
        if   0.2 < prediction[0+counter][1] < 0.8:
            nevim_80 = nevim_80 + 1
        if   0.1 > prediction[0+counter][1]:
            chyba_90 = chyba_90 + 1
        if   0.2 > prediction[0+counter][1]:
            chyba_80 = chyba_80 + 1
        if   0.5 < prediction[0+counter][1]:
            vim_50 = vim_50 + 1
        else:   
            vim_90 = vim_90 + 0    
        counter = counter + 1
        if counter == (fin):
            break
    vim_90_txt = str(vim_90)
    vim_80_txt = str(vim_80)
    nevim_90_txt = str(nevim_90)
    nevim_80_txt = str(nevim_80) 
    chyba_90_txt = str(chyba_90)
    chyba_80_txt = str(chyba_80)
    vim_50_txt = str(vim_50)     
    f.writelines(vim_90_txt)
    f.write("\n")
    f.writelines(vim_80_txt)
    f.write("\n")
    f.writelines(nevim_90_txt)
    f.write("\n")
    f.writelines(nevim_80_txt)
    f.write("\n")
    f.writelines(chyba_90_txt)
    f.write("\n")
    f.writelines(chyba_80_txt)
    f.write("\n")
    f.writelines(vim_50_txt)


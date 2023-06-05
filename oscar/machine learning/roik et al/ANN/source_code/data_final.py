import random
import decimal
import numpy as np
import math
import ast

def cteni(soubor):
    with open (soubor,"r") as data:       
        train_data_pokus = data.readline().rstrip()
        train_data = ast.literal_eval(train_data_pokus)   
    
        train_labels_pokus = data.readline().rstrip()
        train_labels = ast.literal_eval(train_labels_pokus)

        igen_values_pokus = data.readline().rstrip()
        igen_values = ast.literal_eval(igen_values_pokus)
        return [train_labels,train_data,igen_values]


soubor_1 = cteni("vyhodnoceni_15_1.txt")
train_data_1 = soubor_1[1]
train_labels_1 = soubor_1[0]
igen_values_1 = soubor_1[2]
#print(train_data_1)
#print(train_labels_1)
#print(igen_values_1)

soubor_2 = cteni("vyhodnoceni_15_2.txt")
train_data_2 = soubor_2[1]
train_labels_2 = soubor_2[0]
igen_values_2 = soubor_2[2]
#print(train_data_2)
#print(train_labels_2)
#print(igen_values_2)

soubor_3 = cteni("vyhodnoceni_15_3.txt")
train_data_3 = soubor_3[1]
train_labels_3 = soubor_3[0]
igen_values_3 = soubor_3[2]

soubor_4 = cteni("vyhodnoceni_15_4.txt")
train_data_4 = soubor_4[1]
train_labels_4 = soubor_4[0]
igen_values_4 = soubor_4[2]

soubor_5 = cteni("vyhodnoceni_15_5.txt")
train_data_5 = soubor_5[1]
train_labels_5 = soubor_5[0]
igen_values_5 = soubor_5[2]

soubor_6 = cteni("vyhodnoceni_15_6.txt")
train_data_6 = soubor_6[1]
train_labels_6 = soubor_6[0]
igen_values_6 = soubor_6[2]

soubor_7 = cteni("vyhodnoceni_15_7.txt")
train_data_7 = soubor_7[1]
train_labels_7 = soubor_7[0]
igen_values_7 = soubor_7[2]

soubor_8 = cteni("vyhodnoceni_15_8.txt")
train_data_8 = soubor_8[1]
train_labels_8 = soubor_8[0]
igen_values_8 = soubor_8[2]

soubor_9 = cteni("vyhodnoceni_15_9.txt")
train_data_9 = soubor_9[1]
train_labels_9 = soubor_9[0]
igen_values_9 = soubor_9[2]

soubor_10 = cteni("vyhodnoceni_15_10.txt")
train_data_10 = soubor_10[1]
train_labels_10 = soubor_10[0]
igen_values_10 = soubor_10[2]


train_data = []
train_labels = []
igen_values = []


def final_sample(train_data_x,train_labels_x,igen_values_x):
    t = 0
    while t < len(train_labels_1):
        train_data.append(train_data_x[0+t])
        train_labels.append(train_labels_x[0+t])
        igen_values.append(igen_values_x[0+t])
        t = t + 1

final_sample(train_data_1,train_labels_1,igen_values_1)
final_sample(train_data_2,train_labels_2,igen_values_2)
final_sample(train_data_3,train_labels_3,igen_values_3)
final_sample(train_data_4,train_labels_4,igen_values_4)
final_sample(train_data_5,train_labels_5,igen_values_5)
final_sample(train_data_6,train_labels_6,igen_values_6)
final_sample(train_data_7,train_labels_7,igen_values_7)
final_sample(train_data_8,train_labels_8,igen_values_8)
final_sample(train_data_9,train_labels_9,igen_values_9)
final_sample(train_data_10,train_labels_10,igen_values_10)
#print(train_data)
#print(train_labels)
#print(igen_values)

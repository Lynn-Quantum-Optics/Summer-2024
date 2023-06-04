import random
import decimal
import numpy as np
import math
import ast


def cteni(soubor):
    with open (soubor,"r") as data:       
        train_data_pokus = data.readline().rstrip()
        train_data = ast.literal_eval(train_data_pokus) 

        test_data_pokus = data.readline().rstrip()
        test_data = ast.literal_eval(test_data_pokus)  
    
        train_labels_pokus = data.readline().rstrip()
        train_labels = ast.literal_eval(train_labels_pokus)

        test_labels_pokus = data.readline().rstrip()
        test_labels = ast.literal_eval(test_labels_pokus)
        return [train_data,test_data,train_labels,test_labels]


soubor_1 = cteni("data_uceni_15_1.txt")
train_data_1 = soubor_1[0]
test_data_1 = soubor_1[1]
train_labels_1 = soubor_1[2]
test_labels_1 = soubor_1[3]
#print(train_data_1)
#print(test_data_1)
#print(train_labels_1)
#print(test_labels_1)


soubor_2 = cteni("data_uceni_15_2.txt")
train_data_2 = soubor_2[0]
test_data_2 = soubor_2[1]
train_labels_2 = soubor_2[2]
test_labels_2 = soubor_2[3]

soubor_3 = cteni("data_uceni_15_3.txt")
train_data_3 = soubor_3[0]
test_data_3 = soubor_3[1]
train_labels_3 = soubor_3[2]
test_labels_3 = soubor_3[3]

soubor_4 = cteni("data_uceni_15_4.txt")
train_data_4 = soubor_4[0]
test_data_4 = soubor_4[1]
train_labels_4 = soubor_4[2]
test_labels_4 = soubor_4[3]

soubor_5 = cteni("data_uceni_15_5.txt")
train_data_5 = soubor_5[0]
test_data_5 = soubor_5[1]
train_labels_5 = soubor_5[2]
test_labels_5 = soubor_5[3]

soubor_6 = cteni("data_uceni_15_6.txt")
train_data_6 = soubor_6[0]
test_data_6 = soubor_6[1]
train_labels_6 = soubor_6[2]
test_labels_6 = soubor_6[3]

soubor_7 = cteni("data_uceni_15_7.txt")
train_data_7 = soubor_7[0]
test_data_7 = soubor_7[1]
train_labels_7 = soubor_7[2]
test_labels_7 = soubor_7[3]

soubor_8 = cteni("data_uceni_15_8.txt")
train_data_8 = soubor_8[0]
test_data_8 = soubor_8[1]
train_labels_8 = soubor_8[2]
test_labels_8 = soubor_8[3]

soubor_9 = cteni("data_uceni_15_9.txt")
train_data_9 = soubor_9[0]
test_data_9 = soubor_9[1]
train_labels_9 = soubor_9[2]
test_labels_9 = soubor_9[3]

soubor_10 = cteni("data_uceni_15_10.txt")
train_data_10 = soubor_10[0]
test_data_10 = soubor_10[1]
train_labels_10 = soubor_10[2]
test_labels_10 = soubor_10[3]


train_data = []
train_labels = []
test_data = []
test_labels = []

def final_sample(train_data_x,train_labels_x,test_data_x,test_labels_x):
    t = 0
    while t < len(train_labels_1):
        train_data.append(train_data_x[0+t])
        train_labels.append(train_labels_x[0+t])
        t = t + 1
    h = 0
    while h < len(test_data_1):
        test_data.append(test_data_x[0+h])
        test_labels.append(test_labels_x[0+h])        
        h = h + 1

final_sample(train_data_1,train_labels_1,test_data_1,test_labels_1)
print("1")
final_sample(train_data_2,train_labels_2,test_data_2,test_labels_2)
print("2")
final_sample(train_data_3,train_labels_3,test_data_3,test_labels_3)
print("3")
final_sample(train_data_4,train_labels_4,test_data_4,test_labels_4)
print("4")
final_sample(train_data_5,train_labels_5,test_data_5,test_labels_5)
print("5")
final_sample(train_data_6,train_labels_6,test_data_6,test_labels_6)
print("6")
final_sample(train_data_7,train_labels_7,test_data_7,test_labels_7)
print("7")
final_sample(train_data_8,train_labels_8,test_data_8,test_labels_8)
print("8")
final_sample(train_data_9,train_labels_9,test_data_9,test_labels_9)
print("9")
final_sample(train_data_10,train_labels_10,test_data_10,test_labels_10)
print("10")

#print(test_data[1])
#print(train_data[1])
#print(test_labels[1])
#print(train_labels[1])


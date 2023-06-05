import random
import decimal
import numpy as np
import math
import ast

def Read_data(file_name):
    test_data_1_pokus = []     
    with open (file_name,"r") as data:       
        line = data.readline().rstrip()                               
        n=line.split()
        value = [float(n[0]),float(n[1]),float(n[2]),float(n[3])]
        test_data_1_pokus.append(value)
        while line:
            line = data.readline().rstrip()
            n=line.split()
            if n == []:
                break             
            value = [float(n[0]),float(n[1]),float(n[2]),float(n[3])]
            test_data_1_pokus.append(value)
    return test_data_1_pokus


test_data_1 = Read_data("ccDepol075.dat")
#print(test_data_1)


p = 0

test_data = []
test_labels = []
prediction = []
colectibility = []
number = 0
spatnych = 0

while len(test_data_1) > number:
    print("Dalsi")
    print(number)
    projekce_HH = test_data_1[0+number][0]
    print(projekce_HH)
    projekce_VV = test_data_1[0+number][1]
    print(projekce_VV)
    projekce_HV = test_data_1[0+number][2]
    print(projekce_HV)
    projekce_AA = test_data_1[0+number][3]
    print(projekce_AA)
    projekce_DD = test_data_1[0+number][3]
    p_H = 0.5

 

    if projekce_DD >= projekce_AA:
        mik = 16*p_H*(1-p_H)*(projekce_HH*projekce_VV)**(1/2)+4*projekce_DD
#    print(f"ni: {mik}")
    else:
        mik = 16*p_H*(1-p_H)*(projekce_HH*projekce_VV)**(1/2)+4*projekce_AA     

    w_ro = ((mik + p_H**2*(1-2*projekce_HH)+(1-p_H)**2*(1-2*projekce_VV)+2*p_H*(1-p_H)*(1-2*projekce_HV)-1))/2
    

    if type(w_ro) == type(test_data_1[0+number][3]):
        colectibility.append(w_ro)
        test_data_1[0+number].append(projekce_DD)        
        test_data.append(test_data_1[0+number])
        print (w_ro)
        if w_ro < 0:
            prediction.append(0)

        else:
            prediction.append(1)

        if p > 1/3:
            test_labels.append(0)
        else:
            test_labels.append(1)   

        number = number +1
    else:
        number = number +1
        spatnych = spatnych +1
        continue

next = 0
chyba_1 = 0
chyba_2 = 0
dobre = 0

while len(test_labels) > next:
    if 1 == test_labels[0+next] and prediction[0+next] != test_labels [0+next]:
        chyba_1 = chyba_1 + 1 
    elif 0 == test_labels[0+next] and prediction[0+next] != test_labels [0+next]:
        chyba_2 = chyba_2 + 1
    elif 0 == test_labels[0+next] and prediction[0+next] == test_labels [0+next]:
        dobre = dobre + 1
    elif 1 == test_labels[0+next] and prediction[0+next] == test_labels [0+next]:
        dobre = dobre + 1
    next = next +1     


print('dobre')
print(dobre)

print('chyba_1')
print(chyba_1)

print('chyba_2')
print(chyba_2)

print("spatnyh")
print(spatnych)

print(test_data[0])
print(len(test_data))
print(len(test_labels))


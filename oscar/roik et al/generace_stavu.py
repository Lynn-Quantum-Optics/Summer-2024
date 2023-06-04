import random
import decimal
import numpy as np
import math



#p = [0,0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
p = [0]
p_add = 0
x = 0
while x < 100000:     
    p_add = p_add + 0.00001
    p.append(p_add)                 
    x = x + 1



fail= []
next = 0
train_data = []
train_labels = []
prediction = []
igen_values = []
chyba_1 = 0
chyba_2 = 0
dobre = 0

colectibility= []

next = 0
train_data = []
while len(p) > next:

    projekce_HH_smeska = (p[0+next])**2*0.01
    projekce_HV_smeska = (p[0+next])**2*0.483
    projekce_VV_smeska = (p[0+next])**2*0.032
    projekce_DD_smeska = (p[0+next])**2*0.008
    projekce_AA_smeska = (p[0+next])**2*-0.022

    projekce_HH_bell = (1-(p[0+next])**2)*0.242
    projekce_HV_bell = (1-(p[0+next])**2)*0.241
    projekce_VV_bell = (1-(p[0+next])**2)*0.245
    projekce_DD_bell = (1-(p[0+next])**2)*0.225
    projekce_AA_bell = (1-(p[0+next])**2)*0.242
    

    projekce_HH = projekce_HH_smeska + projekce_HH_bell
    projekce_HV = projekce_HV_smeska + projekce_HV_bell
    projekce_VV = projekce_VV_smeska + projekce_VV_bell
    projekce_DD = projekce_DD_smeska + projekce_DD_bell
    projekce_AA = projekce_AA_smeska + projekce_AA_bell



    H_5 = [projekce_HH.real,projekce_VV.real,projekce_HV.real,projekce_DD.real,projekce_AA.real]

    train_data.append(H_5)
    p_H = 0.5

    if projekce_DD >= projekce_AA:
        mik = 16*p_H*(1-p_H)*(projekce_HH*projekce_VV)**(1/2)+4*projekce_DD
#    print(f"ni: {mik}")
    else:
        mik = 16*p_H*(1-p_H)*(projekce_HH*projekce_VV)**(1/2)+4*projekce_AA     

    w_ro = ((mik + p_H**2*(1-2*projekce_HH)+(1-p_H)**2*(1-2*projekce_VV)+2*p_H*(1-p_H)*(1-2*projekce_HV)-1))/2

    colectibility.append(w_ro)
    print(w_ro)

    if w_ro < 0:
        prediction.append(1)


    else:
        prediction.append(0)


    if p[0+next] < 1/3:
        train_labels.append(0)
    else:
        train_labels.append(1) 


    next = next +1



next = 0
while len(train_labels)>next:
    if 1 == train_labels[0+next] and prediction[0+next] != train_labels [0+next]:
        chyba_1 = chyba_1 + 1 
    elif 0 == train_labels[0+next] and prediction[0+next] != train_labels [0+next]:
        chyba_2 = chyba_2 + 1
    elif 0 == train_labels[0+next] and prediction[0+next] == train_labels [0+next]:
        dobre = dobre + 1
    elif 1 == train_labels[0+next] and prediction[0+next] == train_labels [0+next]:
        dobre = dobre + 1
    next = next +1 

print('dobre')
print(dobre)

print('chyba_1')
print(chyba_1)

print('chyba_2')
print(chyba_2)





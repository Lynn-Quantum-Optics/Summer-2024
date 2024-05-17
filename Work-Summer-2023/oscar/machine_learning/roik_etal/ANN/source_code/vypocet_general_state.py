import random
import decimal
import numpy as np
import math
from numpy.linalg import eig

ale = math.tau
b = math.pi
e = math.e
zero = 0
one = 1


def matrix_generation():
    eList = []
    eList.append(np.random.rand())
    eList.append(np.random.rand()*(1-eList[0]))
    eList.append(np.random.rand()*(1-eList[0]-eList[1]))
    eList.append(np.random.rand()*(1-eList[0]-eList[1]-eList[2]))
    eList = np.random.permutation(eList)
    A = eList[0]
    B = eList[1]
    C = eList[2]
    D = eList[3]
    N = A + B + C + D
    if N == 0:
        eList = []
        eList.append(np.random.rand())
        eList.append(np.random.rand()*(1-eList[0]))
        eList.append(np.random.rand()*(1-eList[0]-eList[1]))
        eList.append(np.random.rand()*(1-eList[0]-eList[1]-eList[2]))
        eList = np.random.permutation(eList)
        A = eList[0]
        B = eList[1]
        C = eList[2]
        D = eList[3]
        N_new =  A + B + C + D              
        M = np.matrix([[A/N_new,0,0,0],[0,B/N_new,0,0],[0,0,C/N_new,0],[0,0,0,D/N_new]])    
    else:
        M = np.matrix([[A/N,0,0,0],[0,B/N,0,0],[0,0,C/N,0],[0,0,0,D/N]])    
    matrix_trace = M.trace()
#    print (M)
#    print (matrix_trace)
    return M

def unitary_transform(p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_10,p_11,p_12,p_13,p_14,p_15,p_16):
    Unitary = np.matrix([[p_1,p_2,p_3,p_4],[p_5,p_6,p_7,p_8],[p_9,p_10,p_11,p_12],[p_13,p_14,p_15,p_16]])    
    return Unitary


def projection(p_1,p_2,x,m,phi):
    pp = np.kron(p_1,x)
    PP = np.kron(pp,p_2)
    result_PP= phi @ PP
    r_PP = result_PP.trace()
    cor_pp = np.kron(p_1,m)
    cor_PP = np.kron(cor_pp,p_2)
    cor_res_PP = phi @ cor_PP
    cor_r_PP = cor_res_PP.trace()
    #print(r_VV)
    #print(cor_r_VV)
    fin_r_PP = r_PP/cor_r_PP
    return  fin_r_PP

fail= []
next = 0
train_data = []
train_labels = []
train_data_3 = []
train_data_5 = []
train_data_6 = []
train_data_12 = []
train_data_15 = []
train_data_21 = []
train_data_24 = []

while len(train_data_3) < 400000:
    matrix = matrix_generation()
    #print(matrix)

    ########## 1 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_1 =    unitary_transform(one,zero,zero,zero,zero,one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4)
    #print(unitary_1)
    Unitary_1_herm = unitary_1.transpose().conjugate()

    ########## 2 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_2 = unitary_transform(one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4,zero,zero,zero,zero,one)
#print(unitary_2)
    Unitary_2_herm = unitary_2.transpose().conjugate()

    ########## 3 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_3 = unitary_transform(u_1,u_2,zero,zero,u_3,u_4,zero,zero,zero,zero,one,zero,zero,zero,zero,one)
    #print(unitary_3)
    Unitary_3_herm = unitary_3.transpose().conjugate()

    ########## 4 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_4 = unitary_transform(one,zero,zero,zero,zero,one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4)
    #print(unitary_4)
    Unitary_4_herm = unitary_4.transpose().conjugate()

    ########## 5 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_5 = unitary_transform(one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4,zero,zero,zero,zero,one)
    #print(unitary_5)
    Unitary_5_herm = unitary_5.transpose().conjugate()

    ########## 6 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_6 = unitary_transform(one,zero,zero,zero,zero,one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4)
    #print(unitary_6)

    Unitary_fin = unitary_1 @ unitary_2 @ unitary_3 @ unitary_4 @ unitary_5 @ unitary_6 
    Unitary_fin_herm = Unitary_fin.transpose().conjugate()

    resoult = np.array(Unitary_fin @ matrix @ Unitary_fin_herm)
    #print(resoult) 
    #print("Ahoj")
    check = np.array([[resoult.item(0, 0),resoult.item(1, 0),resoult.item(0, 2),resoult.item(1, 2)],[resoult.item(0, 1),resoult.item(1, 1),resoult.item(0, 3),resoult.item(1, 3)],[resoult.item(2, 0),resoult.item(3, 0),resoult.item(2, 2),resoult.item(3, 2)],[resoult.item(2, 1),resoult.item(3, 1),resoult.item(2, 3),resoult.item(3, 3)]])

    #print(check)
    resoult2 = np.array([[resoult.item(0, 0),resoult.item(0,2),resoult.item(0, 1),resoult.item(0, 3)],[resoult.item(2, 0),resoult.item(2, 2),resoult.item(2,1),resoult.item(2,3)],[resoult.item(1,0),resoult.item(1,2),resoult.item(1,1),resoult.item(1,3)],[resoult.item(3,0),resoult.item(3,2),resoult.item(3,1),resoult.item(3,3)]])

    values , vectors = eig(check)
    #print(values)
    #print(vectors)

    values.sort()



########################trin data#################################
    h = [[1,0],[0,0]]
    v = [[0,0],[0,1]]    
    d = [[1/2,1/2],[1/2,1/2]]
    r = [[1/2,1j/2],[-1j/2,1/2]]
    l = [[1/2,-1j/2],[1j/2,1/2]]
    a = [[1/2,-1/2],[-1/2,1/2]]
    x = [[0,0,0,0],[0,1/2,-1/2,0],[0,-1/2,1/2,0],[0,0,0,0]]
    phi = np.kron(resoult,resoult2)
    m = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] 

    projekce_HH =   projection(h,h,x,m,phi)
    projekce_HV =   projection(h,v,x,m,phi)
    projekce_VH =   projection(v,h,x,m,phi)
    projekce_VV =   projection(v,v,x,m,phi)
    projekce_DD =   projection(d,d,x,m,phi)
    projekce_DA =   projection(d,a,x,m,phi)
    projekce_AD =   projection(a,d,x,m,phi)
    projekce_AA =   projection(a,a,x,m,phi)   
    projekce_RR =   projection(r,r,x,m,phi)
    projekce_RL =   projection(r,l,x,m,phi)
    projekce_LR =   projection(l,r,x,m,phi)
    projekce_LL =   projection(l,l,x,m,phi)
    projekce_HD =   projection(h,d,x,m,phi)
    projekce_HA =   projection(h,a,x,m,phi)
    projekce_VD =   projection(v,d,x,m,phi)
    projekce_VA =   projection(v,a,x,m,phi)
    projekce_HR =   projection(h,r,x,m,phi)
    projekce_HL =   projection(h,l,x,m,phi)
    projekce_VR =   projection(v,r,x,m,phi)
    projekce_VL =   projection(v,l,x,m,phi)
    projekce_DR =   projection(d,r,x,m,phi)
    projekce_DL =   projection(d,l,x,m,phi)
    projekce_AR =   projection(a,r,x,m,phi)
    projekce_AL =   projection(a,l,x,m,phi)
#24#
    projekce_RH =   projection(r,h,x,m,phi)
    projekce_RD =   projection(r,d,x,m,phi)
    projekce_LV =   projection(l,v,x,m,phi)
#3#

    projekce_AH =   projection(a,h,x,m,phi)
    projekce_AV =   projection(a,v,x,m,phi)
    projekce_DV =   projection(d,v,x,m,phi)
    projekce_RV =   projection(r,v,x,m,phi)
    projekce_RA =   projection(r,a,x,m,phi)
    projekce_LH =   projection(l,h,x,m,phi)
    projekce_LA =   projection(l,a,x,m,phi)
    projekce_LD =   projection(l,d,x,m,phi)
    projekce_DH =   projection(d,h,x,m,phi)
#9#



#36    H = [projekce_HH.real,projekce_HV.real,projekce_VH.real,projekce_VV,projekce_DD,projekce_DA,projekce_AD,projekce_AA,projekce_RR,projekce_RL,projekce_LR,projekce_HD.real,projekce_HA.real,projekce_VA,projekce_AL,projekce_HR.real,projekce_HL.real,projekce_VR,projekce_VL,projekce_DR,projekce_DL,projekce_AR,projekce_LL,projekce_RH.real,projekce_RD,projekce_LV,projekce_VD,projekce_AH.real,projekce_AV,projekce_DV,projekce_RV,projekce_RA,projekce_LH.real,projekce_LA,projekce_LD,projekce_DH.real]

    H_24 = [projekce_HH.real,projekce_VV,projekce_HV.real,projekce_DD,projekce_RR,projekce_VA,projekce_RH.real,projekce_HL.real,projekce_RD,projekce_LV,projekce_AA,projekce_LL,projekce_VH.real,projekce_DA,projekce_AD,projekce_RL,projekce_LR,projekce_HD.real,projekce_HA.real,projekce_VD,projekce_VL,projekce_DR,projekce_DL,projekce_AR]
#24new    H = [projekce_HV.real,projekce_VL,projekce_RD,projekce_LR,projekce_DA,projekce_AV,projekce_HL.real,projekce_VV,projekce_RH.real,projekce_LL,projekce_DV,projekce_AA,projekce_HA.real,projekce_VA,projekce_RV,projekce_LD,projekce_DH.real,projekce_AL,projekce_HH.real,projekce_VR,projekce_RR,projekce_LA,projekce_DD,projekce_AH.real]

    H_21 = [projekce_HH.real,projekce_HV.real,projekce_VV,projekce_AV,projekce_AH.real,projekce_DV,projekce_DH.real,projekce_RV,projekce_RH.real,projekce_LV,projekce_LH.real,projekce_RR,projekce_LR,projekce_LL,projekce_DL,projekce_DR,projekce_AL,projekce_AR,projekce_AA,projekce_DA,projekce_DD]

    H_15 = [projekce_DD.real,projekce_AA,projekce_DL,projekce_AR,projekce_DH.real,projekce_AV,projekce_LL,projekce_RR,projekce_LH.real,projekce_RV,projekce_HH.real,projekce_VV,projekce_DR,projekce_DV,projekce_LV]

#    H_12 = [projekce_HH.real,projekce_VV,projekce_HV.real,projekce_DD,projekce_RR,projekce_VA,projekce_RH.real,projekce_HL.real,projekce_RD,projekce_LV,projekce_AA,projekce_LL]
    H_12 = [projekce_DD.real,projekce_AA,projekce_DL,projekce_AR,projekce_DH.real,projekce_AV,projekce_LL.real,projekce_RR,projekce_LH.real,projekce_RV,projekce_HH.real,projekce_VV]


    H_6 = [projekce_HH.real,projekce_VV.real,projekce_HV.real,projekce_DD.real,projekce_RR,projekce_LL]
#6new    H = [projekce_HV.real,projekce_VL,projekce_RD,projekce_LR,projekce_DA,projekce_AR]


    H_5 = [projekce_HH.real,projekce_VV.real,projekce_HV.real,projekce_DD.real,projekce_AA.real]


#4new    H = [projekce_HV.real,projekce_RL,projekce_DA,projekce_RD]
#    H = [projekce_HH.real,projekce_VV.real,projekce_HV.real,projekce_DD.real]


    H_3 = [projekce_HH.real,projekce_VV.real,projekce_HV.real]
#3new    H = [projekce_HV.real,projekce_RL,projekce_DA]

    train_data_3.append(H_3)
    train_data_5.append(H_5)
    train_data_6.append(H_6)
    train_data_12.append(H_12)
    train_data_15.append(H_15)
    train_data_24.append(H_24)
    train_data_21.append(H_21)
    

    if values[:1] < 0:
        train_labels.append(0)

    else:
        train_labels.append(1)



#    if -0.05 < values[:1] < 0.05:
#        train_data.append(H)
#        if values[:1] < 0:
#            train_labels.append(0)
#
#        else:
#            train_labels.append(1)
#
#    else:
#        fail.append(H)

    next = next + 1
    print(next)
#    print(values)                 
#print(train_labels)
#print(train_data)





next = 0
test_data = []
test_labels = []
test_data_3 = []
test_data_5 = []
test_data_6 = []
test_data_12 = []
test_data_15 = []
test_data_24 = []
test_data_21 = []


while len(test_data_3) < 40000:
    matrix = matrix_generation()
    #print(matrix)

    ########## 1 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_1 =    unitary_transform(one,zero,zero,zero,zero,one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4)
    #print(unitary_1)
    Unitary_1_herm = unitary_1.transpose().conjugate()

    ########## 2 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_2 = unitary_transform(one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4,zero,zero,zero,zero,one)
#print(unitary_2)
    Unitary_2_herm = unitary_2.transpose().conjugate()

    ########## 3 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_3 = unitary_transform(u_1,u_2,zero,zero,u_3,u_4,zero,zero,zero,zero,one,zero,zero,zero,zero,one)
    #print(unitary_3)
    Unitary_3_herm = unitary_3.transpose().conjugate()

    ########## 4 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_4 = unitary_transform(one,zero,zero,zero,zero,one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4)
    #print(unitary_4)
    Unitary_4_herm = unitary_4.transpose().conjugate()

    ########## 5 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_5 = unitary_transform(one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4,zero,zero,zero,zero,one)
    #print(unitary_5)
    Unitary_5_herm = unitary_5.transpose().conjugate()

    ########## 6 unit ##########
    alpha = random.randint(0,1000)/1000*ale
    phi = random.randint(0,1000)/1000*ale
    ksi = random.randint(0,1000)/1000*ale     
    theta = math.asin((random.randint(0,100000)/100000)**(1/2))

    u_1 = e**(alpha*1j)*e**(phi*1j)*math.cos(theta)
    u_2 = e**(alpha*1j)*e**(ksi*1j)*math.sin(theta)
    u_3 = e**(alpha*1j)*-e**-(ksi*1j)*math.sin(theta)
    u_4 = e**(alpha*1j)*e**-(phi*1j)*math.cos(theta)

    unitary_6 = unitary_transform(one,zero,zero,zero,zero,one,zero,zero,zero,zero,u_1,u_2,zero,zero,u_3,u_4)
    #print(unitary_6)

    Unitary_fin = unitary_1 @ unitary_2 @ unitary_3 @ unitary_4 @ unitary_5 @ unitary_6 
    Unitary_fin_herm = Unitary_fin.transpose().conjugate()

    resoult = np.array(Unitary_fin @ matrix @ Unitary_fin_herm)
    #print(resoult) 
    #print("Ahoj")
    check = np.array([[resoult.item(0, 0),resoult.item(1, 0),resoult.item(0, 2),resoult.item(1, 2)],[resoult.item(0, 1),resoult.item(1, 1),resoult.item(0, 3),resoult.item(1, 3)],[resoult.item(2, 0),resoult.item(3, 0),resoult.item(2, 2),resoult.item(3, 2)],[resoult.item(2, 1),resoult.item(3, 1),resoult.item(2, 3),resoult.item(3, 3)]])

    #print(check)
    resoult2 = np.array([[resoult.item(0, 0),resoult.item(0,2),resoult.item(0, 1),resoult.item(0, 3)],[resoult.item(2, 0),resoult.item(2, 2),resoult.item(2,1),resoult.item(2,3)],[resoult.item(1,0),resoult.item(1,2),resoult.item(1,1),resoult.item(1,3)],[resoult.item(3,0),resoult.item(3,2),resoult.item(3,1),resoult.item(3,3)]])


    values , vectors = eig(check)
#    print(values)
    #print(vectors)

    values.sort()




########################test data#################################
    h = [[1,0],[0,0]]
    v = [[0,0],[0,1]]    
    d = [[1/2,1/2],[1/2,1/2]]
    r = [[1/2,1j/2],[-1j/2,1/2]]
    l = [[1/2,-1j/2],[1j/2,1/2]]
    a = [[1/2,-1/2],[-1/2,1/2]]
    x = [[0,0,0,0],[0,1/2,-1/2,0],[0,-1/2,1/2,0],[0,0,0,0]]
    phi = np.kron(resoult,resoult2)
    m = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] 

    projekce_HH =   projection(h,h,x,m,phi)
    projekce_HV =   projection(h,v,x,m,phi)
    projekce_VH =   projection(v,h,x,m,phi)
    projekce_VV =   projection(v,v,x,m,phi)
    projekce_DD =   projection(d,d,x,m,phi)
    projekce_DA =   projection(d,a,x,m,phi)
    projekce_AD =   projection(a,d,x,m,phi)
    projekce_AA =   projection(a,a,x,m,phi)   
    projekce_RR =   projection(r,r,x,m,phi)
    projekce_RL =   projection(r,l,x,m,phi)
    projekce_LR =   projection(l,r,x,m,phi)
    projekce_LL =   projection(l,l,x,m,phi)
    projekce_HD =   projection(h,d,x,m,phi)
    projekce_HA =   projection(h,a,x,m,phi)
    projekce_VD =   projection(v,d,x,m,phi)
    projekce_VA =   projection(v,a,x,m,phi)
    projekce_HR =   projection(h,r,x,m,phi)
    projekce_HL =   projection(h,l,x,m,phi)
    projekce_VR =   projection(v,r,x,m,phi)
    projekce_VL =   projection(v,l,x,m,phi)
    projekce_DR =   projection(d,r,x,m,phi)
    projekce_DL =   projection(d,l,x,m,phi)
    projekce_AR =   projection(a,r,x,m,phi)
    projekce_AL =   projection(a,l,x,m,phi)
#24#
    projekce_RH =   projection(r,h,x,m,phi)
    projekce_RD =   projection(r,d,x,m,phi)
    projekce_LV =   projection(l,v,x,m,phi)
#3#

    projekce_AH =   projection(a,h,x,m,phi)
    projekce_AV =   projection(a,v,x,m,phi)
    projekce_DV =   projection(d,v,x,m,phi)
    projekce_RV =   projection(r,v,x,m,phi)
    projekce_RA =   projection(r,a,x,m,phi)
    projekce_LH =   projection(l,h,x,m,phi)
    projekce_LA =   projection(l,a,x,m,phi)
    projekce_LD =   projection(l,d,x,m,phi)
    projekce_DH =   projection(d,h,x,m,phi)
#9#

#    p_H= 0.5

#    mik = 8*p_H*(1-p_H)*(r_HH/cor_r_HH*r_VV/cor_r_VV)**(1/2)+2*r_DD/cor_r_DD
#    print(f"ni: {mik}")

#    w_ro = round((mik + (p_H**2*(1-2*r_HH/cor_r_HH))+(p_H**2*(1-2*r_VV/cor_r_VV))+2*(p_H**2*(1-2*r_HV/cor_r_HV))-1)/2,4)


#36    H = [projekce_HH.real,projekce_HV.real,projekce_VH.real,projekce_VV,projekce_DD,projekce_DA,projekce_AD,projekce_AA,projekce_RR,projekce_RL,projekce_LR,projekce_HD.real,projekce_HA.real,projekce_VA,projekce_AL,projekce_HR.real,projekce_HL.real,projekce_VR,projekce_VL,projekce_DR,projekce_DL,projekce_AR,projekce_LL,projekce_RH.real,projekce_RD,projekce_LV,projekce_VD,projekce_AH.real,projekce_AV,projekce_DV,projekce_RV,projekce_RA,projekce_LH.real,projekce_LA,projekce_LD,projekce_DH.real]

    H_24 = [projekce_HH.real,projekce_VV,projekce_HV.real,projekce_DD,projekce_RR,projekce_VA,projekce_RH.real,projekce_HL.real,projekce_RD,projekce_LV,projekce_AA,projekce_LL,projekce_VH.real,projekce_DA,projekce_AD,projekce_RL,projekce_LR,projekce_HD.real,projekce_HA.real,projekce_VD,projekce_VL,projekce_DR,projekce_DL,projekce_AR]
#24new    H = [projekce_HV.real,projekce_VL,projekce_RD,projekce_LR,projekce_DA,projekce_AV,projekce_HL.real,projekce_VV,projekce_RH.real,projekce_LL,projekce_DV,projekce_AA,projekce_HA.real,projekce_VA,projekce_RV,projekce_LD,projekce_DH.real,projekce_AL,projekce_HH.real,projekce_VR,projekce_RR,projekce_LA,projekce_DD,projekce_AH.real]

    H_21 = [projekce_HH.real,projekce_HV.real,projekce_VV,projekce_AV,projekce_AH.real,projekce_DV,projekce_DH.real,projekce_RV,projekce_RH.real,projekce_LV,projekce_LH.real,projekce_RR,projekce_LR,projekce_LL,projekce_DL,projekce_DR,projekce_AL,projekce_AR,projekce_AA,projekce_DA,projekce_DD]

    H_15 = [projekce_DD.real,projekce_AA,projekce_DL,projekce_AR,projekce_DH.real,projekce_AV,projekce_LL,projekce_RR,projekce_LH.real,projekce_RV,projekce_HH.real,projekce_VV,projekce_DR,projekce_DV,projekce_LV]

#    H_12 = [projekce_HH.real,projekce_VV,projekce_HV.real,projekce_DD,projekce_RR,projekce_VA,projekce_RH.real,projekce_HL.real,projekce_RD,projekce_LV,projekce_AA,projekce_LL]
    H_12 = [projekce_DD.real,projekce_AA,projekce_DL,projekce_AR,projekce_DH.real,projekce_AV,projekce_LL.real,projekce_RR,projekce_LH.real,projekce_RV,projekce_HH.real,projekce_VV]


#    H_6 = [projekce_HH.real,projekce_VV.real,projekce_HV.real,projekce_DD.real,projekce_RR,projekce_LL]
#6new    H = [projekce_HV.real,projekce_VL,projekce_RD,projekce_LR,projekce_DA,projekce_AR]


    H_5 = [projekce_HH.real,projekce_VV.real,projekce_HV.real,projekce_DD.real,projekce_AA.real]


#4new    H = [projekce_HV.real,projekce_RL,projekce_DA,projekce_RD]
#    H = [projekce_HH.real,projekce_VV.real,projekce_HV.real,projekce_DD.real]


    H_3 = [projekce_HH.real,projekce_VV.real,projekce_HV.real]
#3new    H = [projekce_HV.real,projekce_RL,projekce_DA]

#    if -0.05 < values[:1] < 0.05:
#        test_data.append(H)
#        if values[:1] < 0:
#            test_labels.append(0)
#
#        else:
#            test_labels.append(1)
#    else:
#        fail.append(H)


    test_data_3.append(H_3)
    test_data_5.append(H_5)
    test_data_6.append(H_6)
    test_data_12.append(H_12)
    test_data_15.append(H_15)
    test_data_21.append(H_21)
    test_data_24.append(H_24)

    if values[:1] < 0:
        test_labels.append(0)

    else:
        test_labels.append(1)

    next = next + 1
    print(next)                 


#print(test_labels)
#print(test_data)



#print(test_labels)
#print(test_data)
def zapis(soubor,train_data,test_data,train_labels,test_labels): 
    with open(soubor,'w') as f :
        train_data_txt = str(train_data)
        test_data_txt = str(test_data)
        train_labels_txt = str(train_labels)
        test_labels_txt = str(test_labels)       
        f.writelines(train_data_txt)
        f.write("\n")
        f.writelines(test_data_txt)
        f.write("\n")
        f.writelines(train_labels_txt)
        f.write("\n")
        f.writelines(test_labels_txt)
        f.write("\n")

#zapis("data_uceni_3_10.txt",train_data_3,test_data_3,train_labels,test_labels)
#zapis("data_uceni_5_10.txt",train_data_5,test_data_5,train_labels,test_labels)
#zapis("data_uceni_6_10.txt",train_data_6,test_data_6,train_labels,test_labels)
#zapis("data_uceni_12_10.txt",train_data_12,test_data_12,train_labels,test_labels)
zapis("data_uceni_15_1.txt",train_data_15,test_data_15,train_labels,test_labels)
#zapis("data_uceni_24_10.txt",train_data_24,test_data_24,train_labels,test_labels)

#zapis("data_uceni_21_10.txt",train_data_21,test_data_21,train_labels,test_labels)

########################################
########################################
########################################



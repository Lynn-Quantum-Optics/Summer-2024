# file using the source code from roik et al

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

def get_random_rho():
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

    resoult = np.matrix(Unitary_fin @ matrix @ Unitary_fin_herm)

    return resoult
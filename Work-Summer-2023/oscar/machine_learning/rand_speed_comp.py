# file to test the speed of different random generation methods

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random_gen import *
from rho_methods import *

def test_roik_actual(size=1000):
    ''' Fom actual roik paper code: vypocet_general_state.py'''
    import math
    import random
    ale = math.tau ## ale means "but"
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
    #    M = np.matrix([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        matrix_trace = M.trace()
    #    print (M)
    #    print (matrix_trace)
        return M

    def unitary_transform(p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_10,p_11,p_12,p_13,p_14,p_15,p_16):
        Unitary = np.matrix([[p_1,p_2,p_3,p_4],[p_5,p_6,p_7,p_8],[p_9,p_10,p_11,p_12],[p_13,p_14,p_15,p_16]])    
        return Unitary

    def do_calc(): # I put this code into a function so I could check if valid density matrix before advacing counter in while loop
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
        
        
        density_matrix = np.matrix([[resoult.item(0, 0),resoult.item(0,1),resoult.item(0, 2),resoult.item(0,3)],[resoult.item(1, 0),resoult.item(1,1),resoult.item(1, 2),resoult.item(1,3)],[resoult.item(2, 0),resoult.item(2,1),resoult.item(2, 2),resoult.item(2,3)],[resoult.item(3, 0),resoult.item(3,1),resoult.item(3, 2),resoult.item(3,3)]]  )
        ########## I added this part ##########
        if not(is_valid_rho(density_matrix, verbose=True)): 
            print('invalid!!')
        return density_matrix
        ########################################
    ########################################
    t0= time.time()
    i=0
    num_attempts = 0
    purity_ls = []
    while i < size:
        resoult = do_calc()
        if is_valid_rho(resoult):
            i+=1
            purity_ls.append(get_purity(resoult))
        print(i)
        
    tf = time.time()
    print("Time for 100 hurwitz: ", tf-t0)
    print("Time for 1 hurwitz: ", (tf-t0)/size)

    plt.figure(figsize=(10,7))
    plt.title(f'Purity for Paper Roik for {size} states')
    plt.hist(purity_ls, bins=20)
    plt.savefig(f'roik_actual_{size}.pdf')

def test_roik(size=1000):
    t0 = time.time()
    purity_ls = []
    for i in trange(size):
        rho = get_random_roik()
        purity_ls.append(get_purity(rho))
    t1 = time.time()
    print("Time for 100 roik: ", t1-t0)
    print("Time for 1 roik: ", (t1-t0)/size)

    # make histogram
    plt.figure(figsize=(10,7))
    plt.title(f'Purity for Incorrect Phi Definition for {size} states')
    plt.hist(purity_ls, bins=20)
    plt.savefig(f'roik_{size}.pdf')
    
def test_hurwitz(size=1000):
    t0 = time.time()
    purity_ls = []
    for i in trange(size):
        rho = get_random_hurwitz()
        purity_ls.append(get_purity(rho))
    t1 = time.time()
    print("Time for 100 hurwitz: ", t1-t0)
    print("Time for 1 hurwitz: ", (t1-t0)/size)
    plt.figure(figsize=(10,7))
    plt.title(f'Purity for Horowitz (in Roik et al) Definition for {size} states')
    plt.xlabel('Purity')
    plt.ylabel('Count')
    plt.hist(purity_ls, bins=20)
    plt.savefig(f'horowitz_{size}.pdf')

def test_alec(size=1000):
    def random_prob_vector(dim) -> np.ndarray:
        ''' Generates a random vectors of evenly distributed probabilities between 0 and 1.
        '''
        # generate a random vector with n elements that sum to 1
        vec = []
        while len(vec) < dim-1:
            vec.append(np.random.rand()*(1-np.sum(vec)))
        # add last element so vector sum is unity
        vec.append(1-np.sum(vec))
        # randomly permute the vector
        np.random.shuffle(vec)
        return np.array(vec)

    def generate_Aij(dim:int, i:int, j:int, phi:float, psi:float, chi:float) -> np.ndarray:
        ''' These are the fundemental building blocks for the random unitary matrix generation process.
        
        Parameters
        ----------
        dim : int
            The dimension of the unitary matrix being generated.
        i, j : int, int
            The superscript indicies for this matrix.
        phi, psi, chi : float, float, float
            The parameters for this sub-sub unitary.

        Returns
        -------
        np.ndarray of shape (dim, dim)
            The sub unitary matrix A^(i,j)(phi, psi, chi).
        '''
        # start with identity
        A = np.eye(dim, dtype=complex)
        # insert the values for this sub unitary
        A[i,i] = np.cos(phi)*np.exp(1j*psi)
        A[i,j] = np.sin(phi)*np.exp(1j*chi)
        A[j,i] = -np.sin(phi)*np.exp(-1j*chi)
        A[j,j] = np.cos(phi)*np.exp(-1j*psi)
        # return the sub unitary
        return A

    def generate_An(dim:int, n:int, phis:'list[float]', psis:'list[float]', chi:float) -> np.ndarray:
        ''' These are the unitary building blocks for random unitary matrix generation.
        
        Parameters
        ----------
        dim : int
            The dimension of the unitary matrix being generated.
        n : int
            The index of this sub-unitary, starting at 0.
        phis, psis : np.ndarray of shape (n+1,)
            The parameters for each sub-sub unitary.
        chi : float
            The parameter for the first sub-sub unitary.
        
        Returns
        -------
        np.ndarray of shape (dim, dim)
            The sub-unitary matrix A_n(phis, psis, chi).
        '''
        # start with the identity matrix
        U = np.eye(dim, dtype=complex)
        # apply sub unitaries A(0) -> A(n-2)

        for i in range(n+1):
            if i:
                U = U @ generate_Aij(dim, i, n+1, phis[i], psis[i], 0)
            else:
                U = U @ generate_Aij(dim, i, n+1, phis[i], psis[i], chi)
        return U

    def generate_random_unitary(dim:int) -> np.ndarray:
        ''' Generates a random unitary matrix.
        
        Parameters
        ----------
        dim : int
            The dimension of the unitary matrix being generated.
        
        Returns
        -------
        np.ndarray of shape (dim, dim)
            The random unitary matrix.
        '''
        # start with an identity matrix
        U = np.eye(dim, dtype=complex)
        # loop through sub unitaries to apply
        for n in range(dim-1):
            # generate random psis
            psis = np.random.rand(n+1)*2*np.pi
            # generate random chi
            chi = np.random.rand()*2*np.pi
            # generate random phis
            exs = np.random.rand(n+1)
            phis = np.arccos(np.power(exs, 1/(2*np.arange(1,n+2))))
            # generate and apply the sub unitary
            U = U @ generate_An(dim, n, phis, psis, chi)
        # apply overall phase
        U = np.exp(1j*np.random.rand()*2*np.pi)*U
        # return the unitary
        return U

    def random_unit_vector(dim:int) -> np.ndarray:
        ''' Obtain a random vector on the unit sphere. Probability distribution is uniform over the (dim-1)-dimensional surface of the unit sphere.

        Parameters
        ----------
        dim : int
            The dimension of the unit sphere.

        Returns
        -------
        np.ndarray of shape (dim,)
            A random vector on the unit sphere.
        '''
        # use sqrt of probabilities and random phase
        return np.sqrt(random_prob_vector(dim))*np.exp(2j*np.pi*np.random.rand(dim))

    def RDM1(dim:int) -> np.ndarray:
        ''' Generates a random physical density matrix by using a random unitary matrix to generate eigenvectors and assigning each a random random eigenvalue between zero and one.
        
        Parameters
        ----------
        dim : int
            The dimension of the density matrix being generated.
        
        Returns
        -------
        np.ndarray of shape (dim, dim)
            The random density matrix.
        '''
        # generate a random unitary matrix
        U = generate_random_unitary(dim)
        # use the columns of that matrix as eigenvectors
        eigvecs = U.T.reshape(dim, dim, 1)
        # select random eigenvalues
        eigvals = random_unit_vector(dim)**2
        # construct the density matrix from the eigenvectors and eigenvalues
        out = np.zeros((dim,dim), dtype=complex)
        for v, l in zip(eigvecs, eigvals):
            out += l * (v @ v.conj().T)
        # return the density matrix
        return out
    i = 0
    while i< size:
        rho = RDM1(4)
        if is_valid_rho(rho, verbose=True):
            i+=1
        print(i)

# test_roik()
# test_roik_actual()
test_hurwitz(1000)
# test_alec()
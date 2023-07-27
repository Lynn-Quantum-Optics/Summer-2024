''' unitary.py

This file contains helpful functions for handling and generating unitary matricies.
'''

import numpy as np

def generate_Aij(dim:int, i:int, j:int, phi:float, psi:float, chi:float) -> np.ndarray:
    ''' These are the fundemental building blocks for the maziero random unitary matrix generation process.
    
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
    ''' These are the unitary building blocks for maziero random unitary matrix generation.
    
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

def maziero_random_unitary(dim:int) -> np.ndarray:
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

def roik_random_unitary():
    def get_rand_elems():
        alpha = np.random.rand()*2*np.pi
        psi = np.random.rand()*2*np.pi
        chi = np.random.rand()*2*np.pi
        phi = np.arcsin((np.random.randint(0, 100000)/100000)**(1/2))
        return np.matrix([
            [np.e**(psi*1j)*np.cos(phi), np.e**(chi*1j)*np.sin(phi)],
            [-np.e**(-chi*1j)*np.sin(phi), np.e**(-psi*1j)*np.cos(phi)]
        ])*np.e**(alpha*1j)

    # loop and create unitaries from blocks
    unitary_final = np.eye(4, dtype=np.complex)
    for k in range(5, -1, -1): # count down to do multiplicatiom from right to left
        sub_unitary_k = get_rand_elems()
        if k==0 or k==3 or k==5:
            unitary_k = np.matrix(np.block([[np.eye(2), np.zeros((2,2))], [np.zeros((2,2)), sub_unitary_k]]))
        elif k==1 or k==4:
            ul = np.matrix([[1,0], [0, sub_unitary_k[0, 0]]])# upper left
            ur = np.matrix([[0,0], [sub_unitary_k[0, 1], 0]])# upper right
            ll = np.matrix([[0,sub_unitary_k[1,0]], [0, 0]])# lower left
            lr = np.matrix([[sub_unitary_k[1,1], 0], [0, 1]])# lower right
            unitary_k = np.matrix(np.block([[ul, ur],[ll, lr]]))
        else: # k==2
            unitary_k = np.matrix(np.block([[sub_unitary_k, np.zeros((2,2))], [np.zeros((2,2)), np.eye(2)]]))

        # this way correctly builds right to left
        unitary_final =  unitary_k @ unitary_final
    return unitary_final

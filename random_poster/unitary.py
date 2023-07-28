''' unitary.py

This file contains helpful functions for handling and generating unitary matricies.
'''

import numpy as np
from qo_tools import cos, sin, arcsin, expi

def generate_Uij(dim, i, j, phi, psi, chi):
    ''' Fundemental building blocks for maziero's random unitary generation. 
    
    Parameters
    ----------
    dim : int
        The dimension of the output matrix.
    i, j : int, int
        The indicies of this sub unitary. Unlike the paper, these indices start at zero.
    phi, psi, chi : float, float, float
        The parameters for this sub-sub unitary.
    
    Returns
    -------
    np.ndarray of shape (dim,dim)
        The sub unitary matrix U^(i,j)(phi_ij, psi_ij, chi_ij).
    '''
    # start with ones on the diagonal
    u = np.eye(dim, dtype=complex)
    # insert the values as described in the paper
    u[i,i] = cos(phi)*expi(psi)
    u[i,j] = sin(phi)*expi(chi)
    u[j,i] = -np.conj(u[i,j])
    u[j,j] = cos(phi)*expi(-psi)
    # return the sub-sub unitary
    return u

def generate_Un(dim, n, phis, psis, chi):
    ''' Sub-unitary blocks for the Maziero random unitary generation method.

    Parameters
    ----------
    dim : int
        The dimension of the output matrix.
    n : int
        The index of this sub unitary. Unlike Maziero's paper, these are indexed from zero up to dim-2 (inclusive).
    phis, psis : np.ndarray of shape (n+1,)
        The phi and psi parameters for this sub unitary.
    chi : float
        The chi parameter for the first sub-sub unitary.
    
    Returns
    -------
    np.ndarray of shape (dim,dim)
        The sub unitary matrix U_n(phis, psis, chi).
    '''
    # start with the first one (with chi parameter)
    u = generate_Uij(dim, 0, n+1, phis[0], psis[0], chi)
    # loop to add the rest
    for i in range(1, n+1):
        u = generate_Uij(dim, i, n+1, phis[i], psis[i], 0) @ u
    # return the sub unitary
    return u

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
    u = np.eye(dim, dtype=complex)
    # add complex phase alpha
    alpha = np.random.rand()*2*np.pi
    u = u * expi(alpha)
    # loop through sub unitaries to apply
    for n in range(dim-1):
        # generate random psis and chi
        psis = np.random.rand(n+1)*2*np.pi
        chi = np.random.rand()*2*np.pi
        # generate random phis
        exis = np.random.rand(n+1)
        phis = arcsin(np.power(exis, 1/(2*np.arange(1,n+2))))
        # generate and apply the sub unitary
        u = generate_Un(dim, n, phis, psis, chi) @ u
    # return the random unitary
    return u
    
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
    unitary_final = np.eye(4, dtype=complex)
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

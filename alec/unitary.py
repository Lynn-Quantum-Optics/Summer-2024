''' unitary.py

This file contains helpful functions for handling and generating unitary matricies.
'''

import numpy as np

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

def is_unitary(U, tol=1e-10) -> bool:
    ''' Check if a matrix is unitary.
    
    Parameters
    ----------
    U : np.ndarray
        A square matrix to check.
    tol : float
        The tolerance for the check.
    
    Returns
    -------
    bool
        True if U is unitary, False otherwise.
    '''
    if (len(U.shape) != 2) or (U.shape[0] != U.shape[1]):
        raise ValueError('Matrix to check must be square.')
    return np.all(((U @ U.conj().T) - np.eye(4)) < tol)

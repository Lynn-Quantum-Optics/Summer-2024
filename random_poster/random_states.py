import numpy as np
from scipy import linalg as la

from qo_tools import adj, qubit_trace, HS_norm
from simplex import SIMPLEX_METHODS
from unitary import maziero_random_unitary, roik_random_unitary

def random_pure(n, simplex_method='stick'):
    ''' Generate a random pure state in n dimensions.

    Parameters
    ----------
    n : int
        The number of dimensions in the state vector.
    simplex_method : str, optional (default "stick")
        The simplex method to use for generation. Options are:
        - "stick" : stick breaking method
        - "basic" : basic method
        - "roik" : roik method
        - "gauss" : gaussian method
        - "exp" : exponential simplex
    '''
    # check simplex method
    if simplex_method not in SIMPLEX_METHODS:
        raise ValueError(f'invalid simplex method "{simplex_method}"')
    # generate simplex
    x = SIMPLEX_METHODS[simplex_method](n)
    # create state magnitudes
    x = np.array(np.sqrt(x), dtype=complex).reshape(-1,1)
    # add uniform random phase
    x = np.multiply(x, np.exp(2j*np.pi*np.random.rand(n)).reshape(-1,1))
    return x

def random_mixed_unitary(n, unitary_method='maziero', simplex_method='stick'):
    # check simplex method
    if simplex_method not in SIMPLEX_METHODS:
        raise ValueError(f'invalid simplex method "{simplex_method}"')
    # generate simplex
    x = SIMPLEX_METHODS[simplex_method](n)
    # generate unitary
    if unitary_method == 'maziero':
        u = maziero_random_unitary(n)
    elif unitary_method == 'roik':
        assert n == 4, f'Roik unitary method only supports n = 4 (got n={n})'
        u = roik_random_unitary()
    else:
        raise ValueError(f'invalid unitary method "{unitary_method}"')
    # create density matrix
    return u @ np.diag(x) @ adj(u)

def random_mixed_trace(log2n, log2start=None):
    ''' random mixed state generation using the partial-trace method

    starting with a <log2start> qubit random pure state, we trace out <log2n> qubits

    if log2start is left as None, 2*log2n will be used
    '''
    # check input
    if log2start is None: log2start = 2*log2n
    # generate random pure state
    psi_start = random_pure(2**log2start)
    rho_start = psi_start @ adj(psi_start)
    # trace out qubits
    return qubit_trace(rho=rho_start, keep=log2n)

def random_mixed_op(n, dist='exp'):
    ''' random mixed state via overparameterized method

    Parameters
    ----------
    dist : str, optional (default "exp")
        The distribution to sample for the initial parameters. Options are:
        - "exp" : exponential magnitude distribution, uniform phase
        - "uniform" : uniform distribution [0, 1] of magnitudes, uniform phase
        - "normal" : normal distribution of magnitudes, uniform phase
        - "exp re-im" : exponential distribution for both re and im
        - "uniform re-im" : uniform distribution [-1, 1] for both re and im
        - "normal re-im" : normal distribution for both re and im
    '''
    if dist == 'exp':
        # exponential magnitudes
        a = np.random.exponential(size=(n,n))
        # add uniform phase
        a = np.multiply(np.exp(2j*np.pi*np.random.rand(n,n)), a)
    elif dist == 'uniform':
        # uniform magnitudes
        a = np.random.rand(n,n)
        # add uniform phase
        a = np.multiply(np.exp(2j*np.pi*np.random.rand(n,n)), a)
    elif dist == 'normal':
        # normal distribution of magnitudes
        a = np.random.randn(n,n)
        # add uniform phase
        a = np.multiply(np.exp(2j*np.pi*np.random.rand(n,n)), a)
    elif dist == 'exp re-im':
        a = np.random.exponential(size=(n,n)) + 1j*np.random.exponential(size=(n,n))
    elif dist == 'uniform re-im':
        a = (1-2*np.random.rand(n,n)) + 1j*(1-2*np.random.rand(n,n))
    elif dist == 'normal re-im':
        a = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    else:
        raise ValueError(f'invalid distribution "{dist}"')
    # normalize the initializer
    a = a / HS_norm(a)
    # create the operator
    return adj(a) @ a    

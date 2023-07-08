import numpy as np
from scipy import linalg as la
from unitary import generate_random_unitary

# +++ HELPER METHODS FOR GENERATING RANDOM VECTORS +++

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

def random_prob_vector_roik(dim) -> np.ndarray:
    ''' Generates a random vectors of evenly distributed probabilities between 0 and 1.
    '''
    # generate a random vector with n elements that sum to 1
    vec = []
    while len(vec) < dim:
        vec.append(np.random.rand()*(1-np.sum(vec)))
    # add last element so vector sum is unity
    vec = np.array(vec)/np.sum(vec)
    # randomly permute the vector
    np.random.shuffle(vec)
    return vec

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

def random_radius(dim:int) -> float:
    ''' Get a random radius between 0 and 1 in the given dimension.

    The radii returned by this function will satisfy an even probability distribution inside the radius of the unit sphere.
    
    Parameters
    ----------
    dim : int
        The dimension of the unit sphere.

    Returns
    -------
    float
        A random radius between 0 and 1.
    '''
    # the PDF is n*r^(n-1), so the CDF is r^n, which can be analytically inverted so that
    return np.random.rand()**(1/dim)

def random_vector_in_unit_sphere(dim:int) -> np.ndarray:
    ''' Obtain a random vector in the unit sphere. Probability distribution is uniform over the (dim)-dimensional volume of the unit sphere.

    Parameters
    ----------
    dim : int
        The dimension of the unit sphere.
    
    Returns
    -------
    np.ndarray of shape (dim,)
        A random vector in the unit sphere.
    '''
    return random_unit_vector(dim) * random_radius(dim)

# +++ STANDARD METHOD +++

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
    '''
    U = generate_random_unitary(dim)
    rho = np.diag(random_prob_vector_roik(dim))
    return U @ rho @ U.conj().T

# function to generate generalized Gell-Mann matricies

def generate_gell_mann(dim:int) -> np.ndarray:
    ''' Generates the generalized Gell-Mann matrices for a given dimension.
    
    Parameters
    ----------
    dim : int
        The dimension of the matrices being generated.
    
    Returns
    -------
    np.ndarray of shape (dim**2-1, dim, dim)
        The generalized Gell-Mann matrices for this basis.
    '''
    out = []
    # off-diagonal matrices
    for i in range(dim):
        for j in range(i+1,dim):
            x, y = np.zeros((dim,dim), dtype=complex), np.zeros((dim,dim), dtype=complex)
            # real
            x[i,j] = 1
            x[j,i] = 1
            out.append(x)
            # imag
            y[i,j] = -1j
            y[j,i] = 1j
            out.append(y)
    # on-diagonal matrices
    for l in range(dim-1):
        x = np.zeros((dim,dim), dtype=complex)
        # add diagonal entries
        for j in range(l+1):
            x[j,j] = 1
            x[j+1,j+1] = -(l+1)
        # normalize
        x /= np.sqrt((l+1)*(l+2)/2)
        out.append(x)
    return np.array(out, dtype=complex).reshape(-1, dim, dim)

BLOCH_BASIS = generate_gell_mann(4)
BLOCH_VEC_LOWER = np.array(
    [-0.5] * 12 +
    [-np.sqrt((l+1)/(2*(l+2))) for l in range(3)])
BLOCH_VEC_UPPER = np.array(
    [0.5] * 12 +
    [1/np.sqrt(2*(l+1)*(l+2)) for l in range(3)])

def generate_bloch_vector(dim):
    # generate and properly scale the vector
    x = np.random.rand(dim**2-1)
    x *= (BLOCH_VEC_UPPER - BLOCH_VEC_LOWER)
    x += BLOCH_VEC_LOWER
    # return the vector
    return x

def random_su(dim):
    return np.eye(dim)/dim + BLOCH_BASIS.T @ generate_bloch_vector(dim)

def random_rho(dim=4, v=False, cap=1000):
    count = 1
    while count < cap:
        x = random_su(dim)
        evs = np.linalg.eigvals(x)
        if np.all(evs >= 0):
            if v: print('took', count, 'tries')
            return x
        else:
            count += 1

def RDM2(dim:int) -> np.ndarray:
    ''' Generates a random physical density matrix by computing the partial trace of a 2n-qubit pure state.

    Parameters
    ----------
    dim : int
        The dimension of the density matrix being generated. Must be a power of two.

    Returns
    -------
    np.ndarray of shape (dim, dim)
        The random density matrix.
    '''
    # dim is 2^n so
    assert np.log2(dim) % 1 == 0, 'dim must be a power of 2'
    # generate a random physical 2n qubit pure state

    # note: this requires a 2^(2n) sized vector and since dim=2^n, 2^(2n) = dim^2
    # note: randn is a gaussian distribution which provides an even spacing on the 2^(2n)-dimensional hyper-sphere
    # x = np.random.randn(dim**2,1) + 1j*np.random.randn(dim**2,1)
    # x /= np.linalg.norm(x)

    # ALTERNATE METHOD
    # x =  np.sqrt(random_prob_vector(dim**2)) * np.exp(2j*np.pi*np.random.rand(dim**2))
    # x = x.reshape(-1,1)
    x = random_unit_vector(dim**2).reshape(-1,1)

    # calculate density matrix
    rhox = x @ x.conj().T
    # compute the partial trace
    out = np.zeros((dim,dim), dtype=complex)
    for i in range(0, dim**2, dim):
        out += rhox[i:i+dim, i:i+dim]
    return out

# the pauli method relies on evenly spaced vectors in n-dimensional hyper spheres. random direction is achieved by using a normal distribution and then the vector is normalized before being scaled according to a probability distribution derived from the differential volume element with respect to radius.


def random_radius(dim):
    # the PDF is n*r^(n-1), so the CDF is r^n, which can be analytically inverted so that
    return np.random.rand()**(1/dim)

def random_vector_in_unit_sphere(dim):
    x = np.random.randn(dim)
    x /= np.linalg.norm(x)
    x *= random_radius(dim)
    return x


# PAULI METHOD

I = np.eye(2)
SX = np.array([[0,1],[1,0]])
SY = np.array([[0,-1j],[1j,0]])
SZ = np.array([[1,0],[0,-1]])

def generate_pauli_basis(dim:int) -> np.ndarray:
    ''' Generates a basis of the pauli spin matricies (or tensor products thereof) for the given dimension.
    
    Parameters
    ----------
    dim : int
        The dimension of the density matrix being generated. Must be a power of two.
    
    Returns
    -------
    np.ndarray of shape (dim, dim, 2**dim)
        The pauli basis.
    '''
    # get number of qubits for output dim
    n = np.log2(dim)
    assert n % 1 == 0, 'dim must be a power of 2'
    n = int(n)
    # start with pauli spin basis
    basis = [np.copy(I), np.copy(SX), np.copy(SY), np.copy(SZ)]
    # loop to expand the basis
    for _ in range(n-1):
        new_basis = []
        for A in [I, SX, SY, SZ]:
            for B in basis:
                new_basis.append(np.kron(A,B))
        basis = new_basis
    return np.array(basis, dtype=complex).T

TWO_QUBIT_PAULI_BASIS = generate_pauli_basis(4)
ONE_QUBIT_PAULI_BASIS = generate_pauli_basis(2)

def RDM3(dim:int, safe:bool=True) -> np.ndarray:
    ''' Generates a random physical density matrix by computing a linear combination of the Pauli matricies (or a tensor product thereof). This may require multiple tries as it is not guaranteed to be physical (have non-zero eigenvalues).

    Parameters
    ----------
    dim : int
        The dimension of the density matrix being generated. Must be a power of two.
    safe : bool, optional (default=True)
        If true, the output is guaranteed to be a physical density matrix with non-negative eigenvalues. This may be more computationally expensive.
    '''
    # basic cases
    if dim == 4:
        basis = TWO_QUBIT_PAULI_BASIS[:,:,1:]
    elif dim == 2:
        basis = ONE_QUBIT_PAULI_BASIS[:,:,1:]
    else:
        basis = generate_pauli_basis(dim)[:,:,1:]
    
    x = random_vector_in_unit_sphere(dim**2 - 1)
    rho = (np.eye(dim) + basis @ x)/dim

    while True:
        # generate a vector and compute the density matrix
        x = random_vector_in_unit_sphere(dim**2 - 1)
        rho = (np.eye(dim) + basis @ x)/dim
        # check if it is physical
        if safe and np.any(np.linalg.eigvalsh(rho) < 0):
            continue
        else:
            break
    return rho

def FIDE(r1, r2):
    return 1 - np.trace(la.sqrtm(la.sqrtm(r1) @ r2 @ la.sqrtm(r1)))

def adjoint(x):
    return x.conj().T

def is_hermitian(arr, tol=1e-10):
    ''' Check if an array is hermitian. '''
    if (len(arr.shape) != 2) or (arr.shape[0] != arr.shape[1]):
        return False
    return np.all(np.abs(arr - adjoint(arr)) < tol)

def RDM4(dim:int, n:int=2, dist:str='gaussian') -> np.ndarray:
    '''
    '''
    rho = np.zeros((dim,dim), dtype=complex)
    # random probabilities for each pure state
    if dist == 'gaussian':
        probs = np.random.randn(n)
        probs /= np.linalg.norm(probs)
        probs = probs**2
    elif dist == 'uniform':
        probs = np.random.rand(n)
        probs /= np.sum(probs)
    elif dist == 'fair':
        probs = random_prob_vector(dim)
    # try to make a fair probability distribution

    # add pure states with prob p
    for p in probs:
        x = random_unit_vector(dim).reshape(-1,1)
        rho += p * x @ x.conj().T
    return rho

def HSN(A):
    ''' Hilbert-shmitt norm of a hermitian matrix. '''
    return np.sqrt(np.trace(A.conj().T @ A))

def HSD(A, B):
    ''' Hilbert-schmidt distance between two hermitian matrices. '''
    return HSN(A - B)

def RDM5(dim:int, dist='uniform') -> np.ndarray:
    ''' Generates random density matricies using the overparameteized method.

    Parameters
    ----------
    dim : int
        The dimension of the density matrix being generated.
    dist : str, optional (default='uniform')
        The distribution from which to sample parameters. Options are 'uniform', 'uniform non negative', and 'gaussian'.

    Returns
    -------
    np.ndarray of shape (dim, dim)
        The random density matrix.
    '''
    # generate the random parameters
    if dist == 'uniform non negative':
        A = np.random.rand(dim, dim) + 1j*np.random.rand(dim, dim)
    elif dist == 'uniform':
        A = (2*np.random.rand(dim, dim)-1) + 1j*(2*np.random.rand(dim, dim)-1)
    elif dist == 'gaussian':
        A = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    else:
        raise ValueError('dist must be one of "uniform", "uniform non negative", or "gaussian"')
    # normalize
    A /= HSN(A)
    # compute the density matrix
    rho = A @ A.conj().T
    return rho











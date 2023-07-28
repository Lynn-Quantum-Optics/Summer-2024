import numpy as np
from scipy import linalg as la

# basic math functions

expi = lambda x : np.exp(1j*x)
cos = np.cos
sin = np.sin
tan = np.tan
arccos = np.arccos
arcsin = np.arcsin
arctan = np.arctan
sqrt = np.sqrt

# basic helper functions

def Ket(x):
    ''' ket vector of an array '''
    return np.array(x).reshape(-1,1)

def Bra(x):
    ''' bra vector of an array '''
    return np.array(x).reshape(1,-1)

def adj(x):
    ''' adjoint of a matrix '''
    return x.conj().T

def is_ket(x):
    ''' check if a matrix is a ket '''
    return (len(x.shape) == 2) and (x.shape[1] == 1)

def dot(x,y):
    ''' dot product of two kets <x|y> '''
    return (adj(x) @ y).item()

def proj(x,y):
    ''' projection of one ket onto another |<x|y>|^2 '''
    return np.abs(dot(x,y))**2

# wave plates

def rotation_matrix(theta):
    ''' rotation matrix for rotating theta radians counterclockwise '''
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def HWP(theta):
    ''' half wave plate with fast axis at an angle theta from the horizontal '''
    return rotation_matrix(theta) @ np.array([[1,0],[0,-1]]) @ rotation_matrix(-theta)

def QWP(theta):
    ''' quarter wave plate with fast axis at an angle theta from the horizontal '''
    return rotation_matrix(theta) @ np.array([[1,0],[0,1j]]) @ rotation_matrix(-theta)

# matrix helper functions

def two_qubit_partial_transpose(mat):
    ''' Partial transpose on a 4x4 matrix with respect to subsystem 2. '''
    out = np.zeros_like(mat)
    out[0:2,0:2] = mat[0:2,0:2].T
    out[2:4,0:2] = mat[2:4,0:2].T
    out[0:2,2:4] = mat[0:2,2:4].T
    out[2:4,2:4] = mat[2:4,2:4].T
    return out

def is_hermitian(arr, tol=1e-10):
    ''' Check if an array is hermitian. '''
    if (len(arr.shape) != 2) or (arr.shape[0] != arr.shape[1]):
        return False
    else:
        return np.all(np.abs(arr - adj(arr)) < tol)

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
        return False
    else:
        return np.all(np.abs((U @ U.conj().T) - np.eye(4)) < tol)

def is_valid_density(rho, tol=1e-10):
    ''' check if a matrix is a valid density matrix

    Returns
    -------
    bool or str
        True if density matrix is valid, otherwise some string describing how it is invalid.
    '''
    if len(rho.shape) != 2:
        return 'not a matrix'
    elif rho.shape[0] != rho.shape[1]:
        return 'not square'
    # get eigenvalues
    evs = np.linalg.eigvals(rho)
    # go through checks
    if np.any(np.imag(evs) > tol):
        return 'imaginary eigenvalues'
    elif np.any(np.real(evs) < -tol):
        return 'negative eigenvalues'
    elif np.abs(np.real(np.trace(rho)) - 1) > tol:
        return f'trace {np.real(np.trace(rho)):.5f} != 1'
    elif not is_hermitian(rho):
        return 'not hermitian'
    else:
        return True

def qubit_trace(rho, keep):
    ''' partial trace of an n-qubit density matrix 
    
    Parameters
    ----------
    keep : int
        The number of qubits to keep. Only the first <keep> qubits will be kept.
    
    Returns
    -------
    np.ndarray
        The reduced density matrix.
    '''
    # get input size
    in_size = rho.shape[0]
    # get output size
    out_size = 2**keep
    # initialize output matrix
    out = np.zeros((out_size, out_size), dtype=complex)
    # loop over the submatrices of the input
    for i in range(0, in_size, out_size):
        out += rho[i:i+out_size, i:i+out_size]
    # return output
    return out

# pauli spin matricies

ID = np.array(np.eye(2),dtype=complex)
SX = np.array([[0,1],[1,0]], dtype=complex)
SY = np.array([[0,-1j],[1j,0]], dtype=complex)
SZ = np.array([[1,0],[0,-1]], dtype=complex)

# pauli basis

GENERATED_PAULI_BASES = {}

def generate_pauli_basis(nbits):
    ''' generate pauli basis for an nbit-qubit system '''
    # check if basis has already been generated
    if nbits in GENERATED_PAULI_BASES:
        return np.copy(GENERATED_PAULI_BASES[nbits])
    # initialize pauli basis
    basis = [np.copy(ID), np.copy(SX), np.copy(SY), np.copy(SZ)]
    # loop to expand the basis
    for _ in range(nbits-1):
        new_basis = []
        for a in [ID, SX, SY, SZ]:
            for b in basis:
                new_basis.append(np.kron(a,b))
        basis = new_basis
    basis = np.array(basis).T
    # save basis and return
    GENERATED_PAULI_BASES[nbits] = np.copy(basis)
    return basis

# measurement

def bloch_coords(ket_or_rho):
    ''' get the coordinates of a ket on the bloch sphere'''
    if is_ket(ket_or_rho): rho = ket_or_rho @ adj(ket_or_rho)
    else: rho = ket_or_rho
    assert rho.shape == (2,2)
    return np.array([
        np.real(np.trace(SX @ rho)),
        np.real(np.trace(SY @ rho)),
        np.real(np.trace(SZ @ rho))])

def expectation_value(ket_or_rho, operator):
    ''' get the expectation value of a state under some operators
    
    Parameters
    ----------
    ket_or_rho : array
        The state to measure.
    operator : array
        The matrix representation of the operator.
    
    Returns
    -------
    float
        The expectation value of the operator on the state.
    '''
    if is_ket(ket_or_rho):
        return np.real(adj(ket_or_rho) @ operator @ ket_or_rho).item()
    else:
        return np.real(np.trace(operator @ ket_or_rho)).item()

# state statistics

def concurrence(rho):
    ''' Calculates concurrence of a density matrix using the hermitian R=sqrt(sqrt(rho)*rho_tilde*sqrt(rho)) matrix. '''
    sqrt_rho = la.sqrtm(rho)
    rho_tilde = np.kron(SY,SY) @ rho.conj() @ np.kron(SY,SY)
    R = la.sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)
    evs = la.eigvals(R)
    evs = list(evs.real)
    evs.sort(reverse=True)
    return np.max([0,evs[0] - evs[1] - evs[2] - evs[3]])

def concurrence_alt(rho):
    ''' Calculates concurrence from the non-hermitian rho*rho_tilde matrix. '''
    rho_tilde = np.kron(SY,SY) @ rho.conj() @ np.kron(SY,SY)
    M = rho @ rho_tilde
    evs = la.eigvals(M)
    evs = list(evs.real)
    evs.sort(reverse=True)
    return np.max([0, np.sqrt(evs[0]) - np.sqrt(evs[1]) - np.sqrt(evs[2]) - np.sqrt(evs[3])])

def purity(rho):
    ''' get the purity of a state as the trace of it's density matrix squared. '''
    return np.real(np.trace(rho @ rho))

def fidelity(x, y):
    ''' get the fidelity between two states. '''
    return np.real(np.trace(la.sqrtm(la.sqrtm(x) @ y @ la.sqrtm(x))))**2

# hilbert schmidt stuff

def HS_prod(a,b):
    ''' hilbert-schmidt inner product of two hermitian matrices. '''
    return np.real(np.trace(adj(a) @ b))

def HS_norm(a):
    ''' hilbert-schmidt norm of a hermitian matrix. '''
    return np.sqrt(np.real(np.trace(adj(a) @ a)))

def HS_dist(a,b):
    ''' hilbert-schmidt distance between two hermitian matrices. '''
    return HS_norm(a-b)

# one qubit states

H = Ket([1,0])
V = Ket([0,1])
D = Ket([1,1])/np.sqrt(2)
A = Ket([1,-1])/np.sqrt(2)
R = Ket([1,1j])/np.sqrt(2)
L = Ket([1,-1j])/np.sqrt(2)

# two qubit states

HH = np.kron(H,H)
HV = np.kron(H,V)
HD = np.kron(H,D)
HA = np.kron(H,A)
HR = np.kron(H,R)
HL = np.kron(H,L)

VH = np.kron(V,H)
VV = np.kron(V,V)
VD = np.kron(V,D)
VA = np.kron(V,A)
VR = np.kron(V,R)
VL = np.kron(V,L)

AH = np.kron(A,H)
AV = np.kron(A,V)
AD = np.kron(A,D)
AA = np.kron(A,A)
AR = np.kron(A,R)
AL = np.kron(A,L)

DH = np.kron(D,H)
DV = np.kron(D,V)
DD = np.kron(D,D)
DA = np.kron(D,A)
DR = np.kron(D,R)
DL = np.kron(D,L)

RH = np.kron(R,H)
RV = np.kron(R,V)
RD = np.kron(R,D)
RA = np.kron(R,A)
RR = np.kron(R,R)
RL = np.kron(R,L)

LH = np.kron(L,H)
LV = np.kron(L,V)
LD = np.kron(L,D)
LA = np.kron(L,A)
LR = np.kron(L,R)
LL = np.kron(L,L)

# bell states

PHI_PLUS = (HH + VV)/np.sqrt(2)
PHI_MINUS = (HH - VV)/np.sqrt(2)
PSI_PLUS = (HV + VH)/np.sqrt(2)
PSI_MINUS = (HV - VH)/np.sqrt(2)

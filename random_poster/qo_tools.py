import numpy as np

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

def is_hermitian(arr):
    ''' Check if an array is hermitian. '''
    if (len(arr.shape) != 2) or (arr.shape[0] != arr.shape[1]):
        return False
    return np.all(arr == adj(arr))
# pauli spin matricies

ID = np.array(np.eye(2),dtype=complex)
SX = np.array([[0,1],[1,0]], dtype=complex)
SY = np.array([[0,-1j],[1j,0]], dtype=complex)
SZ = np.array([[1,0],[0,-1]], dtype=complex)

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

HH = np.kron(H,H)

# bell states

PHI_PLUS = (HH + VV)/np.sqrt(2)
PHI_MINUS = (HH - VV)/np.sqrt(2)
PSI_PLUS = (HV + VH)/np.sqrt(2)
PSI_MINUS = (HV - VH)/np.sqrt(2)

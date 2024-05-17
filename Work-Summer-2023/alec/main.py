import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# global variables and such

# pauli spin matricies
SX = np.array([[0,1],[1,0]])
SY = np.array([[0,-1j],[1j,0]])
SZ = np.array([[1,0],[0,-1]])

# bell states




# helper functions to do all the calculations

def spin_flip(rho):
    ''' Returns the 'spin-flipped' (tilde) version of a density matrix rho.'''
    # define spin operators
    sy = np.array([[0,-1j],[1j,0]])
    sysy = np.kron(sy,sy)
    # perform spin flipping
    return sysy @ rho.conj() @ sysy

def R_matrix(rho):
    ''' Calculates the Hermitian R matrix for finding concurrence. '''
    sqrt_rho = la.sqrtm(rho)
    rho_tilde = spin_flip(rho)
    return la.sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)

def concurrence(rho):
    ''' Calculates concurrence of a density matrix using R matrix. '''
    R = R_matrix(rho)
    evs = la.eigvals(R)
    evs = list(evs.real)
    evs.sort(reverse=True)
    return np.max([0,evs[0] - evs[1] - evs[2] - evs[3]])

def concurrence_2(rho):
    ''' Calculates concurrence from the non-hermitian rho*rho_tilde matrix. '''
    M = rho @ spin_flip(rho)
    evs = la.eigvals(M)
    evs = list(evs.real)
    evs.sort(reverse=True)
    return np.max([0, np.sqrt(evs[0]) - np.sqrt(evs[1]) - np.sqrt(evs[2]) - np.sqrt(evs[3])])

# random density matrix
rho_rand = np.eye(4)/4

# define the state that we are messing with
pure_state = np.array([0,1,1,0]).reshape(4,1)/np.sqrt(2) # psi_minus
pure_rho = pure_state @ pure_state.conj().T

# calculate the concurrence of the mixed state
C = []
ps = np.linspace(0,1,200)
for p in ps:
    rho = p * pure_rho + (1-p) * rho_rand
    C.append(concurrence(rho))
    if C[-1] > 0 and C[-2] <= 0:
        print('Concurrence is positive for p =', p)
C = np.array(C)

# plot the results
plt.plot(ps, C)
plt.xlabel('p')
plt.ylabel('Concurrence')
plt.show()








from typing import Any
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

################################### Alec's code from quantum.py #################################################

# helper functions for making states and such

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def adjoint(state):
    return state.conj().T

def density_matrix(states_and_probs):
    # reformat the input for the case of a single state
    if isinstance(states_and_probs, np.ndarray):
        states_and_probs = [(states_and_probs, 1)]
    
    # get the dimensions and initialize the density matrix
    dim = np.max(states_and_probs[0][0].shape)
    rho = np.zeros((dim,dim), dtype=complex)
    # add up all the states
    for state, prob in states_and_probs:
        rho += prob * (state @ adjoint(state))
    return rho


# pauli spin matricies

SX = np.array([[0,1],[1,0]])
SY = np.array([[0,-1j],[1j,0]])
SZ = np.array([[1,0],[0,-1]])

# more helper functions for concurrence

def spin_flip(rho):
    ''' Returns the 'spin-flipped' (tilde) version of a density matrix rho.'''
    # perform spin flipping
    sysy = np.kron(SY,SY)
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

def concurrence_alt(rho):
    ''' Calculates concurrence from the non-hermitian rho*rho_tilde matrix. '''
    M = rho @ spin_flip(rho)
    evs = la.eigvals(M)
    evs = list(evs.real)
    evs.sort(reverse=True)
    return np.max([0, np.sqrt(evs[0]) - np.sqrt(evs[1]) - np.sqrt(evs[2]) - np.sqrt(evs[3])])

# one qubit states

ZP = ket([1,0])
ZM = ket([0,1])
XP = ket([1,1])/np.sqrt(2)
XM = ket([1,-1])/np.sqrt(2)
YP = ket([1,1j])/np.sqrt(2)
YM = ket([1,-1j])/np.sqrt(2)

# bell states

PHI_P = ket([1,0,0,1])/np.sqrt(2)
PHI_M = ket([1,0,0,-1])/np.sqrt(2)
PSI_P = ket([0,1,1,0])/np.sqrt(2)
PSI_M = ket([0,1,-1,0])/np.sqrt(2)

# some more specific functions

def get_meas_probs(rho):
    ''' Returns the full set of 36 measurement probabilities. 
    +x, +x; +x, -x; +x, +y; ... ; +y, +x, +y, -x, +y, +y; ... ; -z, -z
    '''
    states = [XP, XM, YP, YM, ZP, ZM]
    probs = []
    for s1 in states:
        for s2 in states:
            s = np.kron(s1, s2)
            p = adjoint(s) @ rho @ s
            probs.append(p.real.item())
    return np.array(probs).reshape(6,6)

def get_meas_expectations(rho):
    ''' Returns measured expectation values for the Pauli operators plus the identity. 
    <1,1>, <1,x>, <1,y>, ... <y,1>, <y,x>, ... <z,z>
    '''
    ops = [np.eye(2), SX, SY, SZ]
    probs = []
    for o1 in ops:
        for o2 in ops:
            o = np.kron(o1, o2)
            probs.append(np.trace(o @ rho).real.item())
    return np.array(probs).reshape(4,4)

def werner_state(eta, pure_state=PSI_M):
    ''' Returns a Werner state with mixing parameter eta. '''
    return eta * pure_state @ adjoint(pure_state) + (1-eta) * np.eye(4)/4

# partial transposition

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
    return np.all(arr == adjoint(arr))

# first six entanglement witnesses (from ricardi)

def W1(theta, meas_exp):
    a, b = np.cos(theta), np.sin(theta)
    # unpack expectation values
    II, XX, YY, ZZ, ZI, IZ = meas_exp[
        [0,1,2,3,3,0],
        [0,1,2,3,0,3]]
    # calculate EW
    return 0.25 * (II + ZZ + (a**2-b**2)*(XX+YY) + 2*a*b*(ZI + IZ))

def W2(theta, meas_exp):
    a, b = np.cos(theta), np.sin(theta)
    # unpack expectation values
    II, XX, YY, ZZ, ZI, IZ = meas_exp[
        [0,1,2,3,3,0],
        [0,1,2,3,0,3]]
    # calculate EW
    return 0.25 * (II - ZZ + (a**2-b**2)*(XX - YY) + 2*a*b*(ZI - IZ))

def W3(theta, meas_exp):
    a, b = np.cos(theta), np.sin(theta)
    # unpack expectation values
    II, XX, YY, ZZ, XI, IX = meas_exp[
        [0,1,2,3,1,0],
        [0,1,2,3,0,1]]
    # calculate EW
    return 0.25 * (II + XX + (a**2-b**2)*(ZZ + YY) + 2*a*b*(XI + IX))

def W4(theta, meas_exp):
    a, b = np.cos(theta), np.sin(theta)
    # unpack expectation values
    II, XX, YY, ZZ, XI, IX = meas_exp[
        [0,1,2,3,1,0],
        [0,1,2,3,0,1]]
    # calculate EW
    return 0.25 * (II - XX + (a**2-b**2)*(ZZ - YY) - 2*a*b*(XI - IX))

def W5(theta, meas_exp):
    a, b = np.cos(theta), np.sin(theta)
    # unpack expectation values
    II, XX, YY, ZZ, YI, IY = meas_exp[
        [0,1,2,3,2,0],
        [0,1,2,3,0,2]]
    # calculate EW
    return 0.25 * (II + YY + (a**2-b**2)*(ZZ + XX) + 2*a*b*(YI + IY))

def W6(theta, meas_exp):
    a, b = np.cos(theta), np.sin(theta)
    # unpack expectation values
    II, XX, YY, ZZ, YI, IY = meas_exp[
        [0,1,2,3,2,0],
        [0,1,2,3,0,2]]
    # calculate EW
    return 0.25 * (II - YY + (a**2-b**2)*(ZZ - XX) - 2*a*b*(YI - IY))

def get_EW_expectations(rho):
    meas_exp = get_meas_expectations(rho)
    witness_expectations = []
    for w in [W1,W2,W3,W4,W5,W6]:
        witness_expectations.append(
            opt.minimize(w, x0=np.pi, args=(meas_exp,), bounds=[(0,2*np.pi)])['fun'])
    return witness_expectations

# function to get the actual EW expectation values

rho = werner_state(0.5, PSI_P)
ew_exp = get_EW_expectations(rho)

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

###################### end of Alec's code ###########################################

########### functions for nonlinear improvements #####################################

def nlW1(theta, meas_exp):
    """
    Turns W1 from Riccardi's 6 and returns the expectation values of the nonlinear components W1, W2, and W3.
    """
    a, b = np.cos(theta), np.sin(theta)
    II, XX, YY, ZZ, ZI, IZ = meas_exp[
        [0,1,2,3,3,0],
        [0,1,2,3,0,3]]
    w1 = 0.125 * (ZI + ZZ + II + IZ + 2*a*b*(ZI + ZZ + II + IZ))
    w2 = 0.125 * (II - IZ - ZI + ZZ + 2*a*b*(IZ - II + ZI - ZZ))
    w3 = 0.25 * (a**2 - b**2) * (XX + YY)
    val = w1*w2 - 0.25 * (abs(w3)**2)
    return val

def nlW2(theta, meas_exp):
    """
    Turns W2 from Riccardi's 6 and returns the expectation values of the nonlinear components W1, W2, and W3.
    """
    a, b = np.cos(theta), np.sin(theta)
    II, XX, YY, ZZ, ZI, IZ = meas_exp[
        [0,1,2,3,3,0],
        [0,1,2,3,0,3]]
    w1 = 0.125 * (ZI - ZZ + II - IZ + 2*a*b*(ZI - ZZ + II - IZ))
    w2 = 0.125 * (II + IZ - ZI - ZZ + 2*a*b*(ZI - II - IZ + ZZ))
    w3 = 0.25 * (a**2 - b**2) * (XX - YY)
    val = w1*w2 - 0.25 * (abs(w3)**2)
    return val

def nlW3(theta, meas_exp):
    """
    Turns W3 from Riccardi's 6 and returns the expectation values of the nonlinear components W1, W2, and W3.
    """
    a, b = np.cos(theta), np.sin(theta)
    II, XX, YY, ZZ, ZI, IZ, IX, ZX, XI = meas_exp[
        [0,1,2,3,3,0,0,3,1],
        [0,1,2,3,0,3,1,1,0]]
    w1 = 0.125 * (II + (a**2 - b**2)*IZ + 2*a*b*IX + ZI + (a**2 - b**2)*ZZ + 2*a*b*ZX)
    w2 = 0.125 * (II - (a**2 - b**2)*IZ + 2*a*b*IX - ZI + (a**2 - b**2)*ZZ - 2*a*b*ZX)
    w3 = 0.25 * (XX + 2*a*b*XI - (a**2 - b**2)*YY)
    val = w1*w2 - 0.25 * (abs(w3)**2)
    return val

def nlW4(theta, meas_exp):
    """
    Turns W4 from Riccardi's 6 and returns the expectation values of the nonlinear components W1, W2, and W3.
    """
    a, b = np.cos(theta), np.sin(theta)
    II, XX, YY, ZZ, ZI, IZ, IX, ZX, XI = meas_exp[
        [0,1,2,3,3,0,0,3,1],
        [0,1,2,3,0,3,1,1,0]]
    w1 = 0.125 * (II + (a**2 - b**2)*IZ + 2*a*b*IX + ZI + (a**2 - b**2)*ZZ + 2*a*b*ZX)
    w2 = 0.125 * (II - (a**2 - b**2)*IZ + 2*a*b*IX - ZI + (a**2 - b**2)*ZZ - 2*a*b*ZX)
    w3 = 0.25 * (-XX - 2*a*b*XI + (a**2 - b**2)*YY)
    val = w1*w2 - 0.25 * (abs(w3)**2)
    return val

def nlW5(theta, meas_exp):
    """
    Turns W5 from Riccardi's 6 and returns the expectation values of the nonlinear components W1, W2, and W3.
    """
    a, b = np.cos(theta), np.sin(theta)
    II, XX, YY, ZZ, ZI, IZ, IY, ZY, YI = meas_exp[
        [0,1,2,3,3,0,0,3,2],
        [0,1,2,3,0,3,2,2,0]]
    w1 = 0.125 * (II + (a**2 - b**2)*IZ + 2*a*b*IY + ZI + (a**2 - b**2)*ZZ + 2*a*b*ZY)
    w2 = 0.125 * (II - (a**2 - b**2)*IZ + 2*a*b*IY - ZI + (a**2 - b**2)*ZZ - 2*a*b*ZY)
    w3 = 0.25 * ((a**2 - b**2)*XX + YY + 2*a*b*YI)
    val = w1*w2 - 0.25 * (abs(w3)**2)
    return val

def nlW6(theta, meas_exp):
    """
    Turns W6 from Riccardi's 6 and returns the inequality value resulting from the expectation values of the nonlinear components W1, W2, and W3.
    """
    a, b = np.cos(theta), np.sin(theta)
    II, XX, YY, ZZ, ZI, IZ, IY, ZY, YI = meas_exp[
        [0,1,2,3,3,0,0,3,2],
        [0,1,2,3,0,3,2,2,0]]
    w1 = 0.125 * (II + (a**2 - b**2)*IZ + 2*a*b*IY + ZI + (a**2 - b**2)*ZZ + 2*a*b*ZY)
    w2 = 0.125 * (II - (a**2 - b**2)*IZ + 2*a*b*IY - ZI + (a**2 - b**2)*ZZ - 2*a*b*ZY)
    w3 = 0.25 * (-(a**2 - b**2)*XX - YY - 2*a*b*YI)
    val = w1*w2 - 0.25 * (abs(w3)**2)
    return val

# outputs < 0 indicate entangled states

def get_nlEW_expectations(rho):
    meas_exp = get_meas_expectations(rho)
    witness_expectations = []
    for w in [nlW1,nlW2,nlW3,nlW4,nlW5,nlW6]:
        witness_expectations.append(opt.minimize(w, x0=np.pi, args=(meas_exp,), bounds=[(0,2*np.pi)])['fun'])
    
    return witness_expectations

# function for returning nonlinear expectation values

rho = werner_state(0.9, PSI_P)
nlew_exp = get_nlEW_expectations(rho)
ew_exp = get_EW_expectations(rho)

# # random density matrix
# rho_rand = np.eye(4)/4

# # define the state that we are messing with
# pure_state = np.array([0,1,-1,0]).reshape(4,1)/np.sqrt(2) # psi_minus
# pure_rho = pure_state @ pure_state.conj().T

# calculate the concurrence of the mixed state
# C = []
# ps = np.linspace(0,1,10000)
# for p in ps:
#     rho = werner_state(p, PSI_P)
#     C.append(concurrence(rho))
#     # if C[-1] > 0 and C[-2] <= 0:
#     #     print('Concurrence is positive for p =', p)
# C = np.array(C)

# exp_values = []
# for p in ps:
#     rho = werner_state(p, PSI_P)
#     ew_exp = get_EW_expectations(rho)
#     exp_values.append(min(ew_exp))
# exp_values = np.array(exp_values)

# nlexp_values = []
# for p in ps:
#     rho = werner_state(p, PSI_P)
#     ew_exp = get_nlEW_expectations(rho)
#     nlexp_values.append(min(ew_exp))
# nlexp_values = np.array(nlexp_values)

# # plot the results
# # plt.plot(ps, C)
# # plt.xlabel('p')
# # plt.ylabel('Concurrence')
# # plt.show()

# plt.plot(ps, exp_values)
# plt.plot(ps, nlexp_values)
# plt.plot(ps, [0 for i in range(10000)])
# plt.xlabel('p')
# plt.ylabel('Minimum Witness value')
# plt.show()
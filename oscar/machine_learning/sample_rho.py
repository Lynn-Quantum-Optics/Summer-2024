# file to hold sample density matrices
import numpy as np

def compute_rho(state):
    ''' Function to compute density matrix from a given state vector. '''
    return state @ np.conjugate(state.reshape((1,4)))

## sample bell states##
PhiP_s = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
PhiP= compute_rho(PhiP_s)

PhiM_s = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).reshape((4,1))
PhiM = compute_rho(PhiM_s)

PsiP_s = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0]).reshape((4,1))
PsiP =  compute_rho(PsiP_s)

PsiM_s = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0]).reshape((4,1))
PsiM =  compute_rho(PsiM_s)

PhiPM_s = 1/np.sqrt(2) *(PhiP_s + PhiM_s)
PhiPM = compute_rho(PhiPM_s)

# Eritas's states
def E_state0(eta, chi): 
    E_state0_s= np.cos(eta)*PhiP_s + np.sin(eta)*np.exp(1j*chi)*PhiM_s 
    return compute_rho(E_state0_s)
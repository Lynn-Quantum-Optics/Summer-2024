# file to hold sample density matrices
import numpy as np
from rho_methods import get_rho

## sample bell states##
PhiP_s = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
PhiP= get_rho(PhiP_s)

PhiM_s = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).reshape((4,1))
PhiM = get_rho(PhiM_s)

PsiP_s = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0]).reshape((4,1))
PsiP =  get_rho(PsiP_s)

PsiM_s = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0]).reshape((4,1))
PsiM =  get_rho(PsiM_s)

PhiPM_s = 1/np.sqrt(2) *(PhiP_s + PhiM_s)
PhiPM = get_rho(PhiPM_s)

# Eritas's states from spring 2023 writeup
def E_state0(eta, chi): 
    E_state0_s= np.cos(eta)*PsiP_s + np.sin(eta)*np.exp(1j*chi)*PsiM_s 
    return get_rho(E_state0_s)

def E_state1(eta, chi):
    E_state1_s= 1/np.sqrt(2) * (np.cos(eta)*(PsiP_s + 1j*PsiM_s) + np.sin(eta)*np.exp(1j*chi)*(PhiP_s + 1j*PhiM_s))
    return get_rho(E_state1_s)
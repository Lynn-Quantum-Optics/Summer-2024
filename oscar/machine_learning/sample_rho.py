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
def get_E0(eta, chi): 
    E_state0_s= np.cos(eta)*PsiP_s + np.sin(eta)*np.exp(1j*chi)*PsiM_s 
    return get_rho(E_state0_s)

def get_E1(eta, chi):
    E_state1_s= 1/np.sqrt(2) * (np.cos(eta)*(PsiP_s + 1j*PsiM_s) + np.sin(eta)*np.exp(1j*chi)*(PhiP_s + 1j*PhiM_s))
    return get_rho(E_state1_s)

## werner states ##
def get_werner_state(p):
    ''' Returns Werner state with parameter p. '''
    return p*PhiP + (1-p)*np.eye(4)/4

## amplitude damped states ##
def get_ads(gamma):
    ''' Returns amplitude damped state with parameter gamma. '''
    return np.array([[.5, 0, 0, .5*np.sqrt(1-gamma)], [0, 0, 0, 0], [0, 0, .5*gamma, 0], [.5*np.sqrt(1-gamma), 0, 0, .5-.5*gamma]])

## sample state to illustrate power of W' over W ##
def get_ex1(phi):
    ''' Returns state with parameter phi. '''
    H = np.array([1,0]).reshape((2,1))
    V = np.array([0,1]).reshape((2,1))
    D = 1/np.sqrt(2) * np.array([1,1]).reshape((2,1))
    A = 1/np.sqrt(2) * np.array([1,-1]).reshape((2,1))
    
    ex1 = np.cos(phi)*np.kron(H,D) - np.sin(phi)*np.kron(V,A)
    return get_rho(ex1)
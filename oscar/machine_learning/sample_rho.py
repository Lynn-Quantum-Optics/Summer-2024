# file to hold sample density matrices
import numpy as np
from rho_methods import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fractions

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
    '''Eritas's state of the form cos(alpha)PsiP + e^(i*beta)*sin(alpha)PsiM'''
    E_state0_s= np.cos(eta)*PsiP_s + np.sin(eta)*np.exp(1j*chi)*PsiM_s 
    return get_rho(E_state0_s)

def get_E1(eta, chi):
    '''Eritas's state of the form 1/sqrt2 * cos(alpha)*(PsiP + iPsiM) + e^(i*beta)*sin(alpha)*(PhiP + iPhiM))'''
    E_state1_s= 1/np.sqrt(2) * (np.cos(eta)*(PsiP_s + 1j*PsiM_s) + np.sin(eta)*np.exp(1j*chi)*(PhiP_s + 1j*PhiM_s))
    return get_rho(E_state1_s)

## sample of different values of eta, chi for E0 and E1
def sample_E():
    '''Samples E0 and E1 for various values of beta and alpha and computes witness values'''
    alpha_ls = np.linspace(0, np.pi/4, 6) # set of alpha values to sample
    beta_ls = np.linspace(0, np.pi/2, 6) # set of beta values to sample

    E0_W_ls = []
    E0_Wp_ls = []
    E1_W_ls = []
    E1_Wp_ls = []

    for beta in beta_ls:
        rho = get_E0(np.pi/3, beta)
        W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho)
        Wp= min(Wp_t1, Wp_t2, Wp_t3)
        E0_W_ls.append(W_min)
        E0_Wp_ls.append(Wp)

    for alpha in alpha_ls:
        rho = get_E1(alpha, np.pi/3)
        W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho)
        Wp= min(Wp_t1, Wp_t2, Wp_t3)
        E1_W_ls.append(W_min)
        E1_Wp_ls.append(Wp)

    # custom tick formatter
    def pi_formatter(x, pos):
        frac = np.round(x / np.pi, decimals=2)
        if np.isclose(frac,0,1e-5):
            return "0"
        else:
            label=fractions.Fraction(frac).limit_denominator()
            num = label.numerator
            denom = label.denominator
            return "$\\frac{%s\pi}{%s}$" %(num, denom) if num!=1 else "$\\frac{\pi}{%s}$" %(denom) if denom!=1 else "$\pi$"
    
    # plot results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(beta_ls, E0_W_ls, label='W')
    ax[0].scatter(beta_ls, E0_Wp_ls, label='W\'')
    ax[0].set_title('$E_0, \\alpha = \\frac{\pi}{3}$')
    ax[0].set_xlabel('$\\beta$')
    ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
    ax[0].set_ylabel('Witness value')
    ax[0].legend()

    ax[1].scatter(alpha_ls, E1_W_ls, label='W')
    ax[1].scatter(alpha_ls, E1_Wp_ls, label='W\'')
    ax[1].set_xlabel('$\\alpha$')
    ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
    ax[1].set_ylabel('Witness value')
    ax[1].set_title('$E_1, \\beta = \\frac{\pi}{3}$')
    ax[1].legend()
    plt.tight_layout()
    plt.suptitle('Witness values for Eritas\'s states')
    plt.savefig('Eritas_witnesses.pdf')
    plt.show()

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

if __name__ == '__main__':
    sample_E()
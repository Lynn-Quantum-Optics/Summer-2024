# file to hold sample density matrices
import numpy as np
from rho_methods import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fractions
from scipy.optimize import curve_fit

## column vectors, standard basis ##
HH = np.array([1,0,0,0]).reshape((4,1))
HV = np.array([0,1,0,0]).reshape((4,1))
VH = np.array([0,0,1,0]).reshape((4,1))
VV = np.array([0,0,0,1]).reshape((4,1))
HH_rho = get_rho(HH)
HV_rho = get_rho(HV)
VH_rho = get_rho(VH)
VV_rho = get_rho(VV)

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
    '''Eritas's state of the form cos(eta)PsiP + e^(i*chi)*sin(eta)PsiM'''
    E_state0_s= np.cos(eta)*PsiP_s + np.sin(eta)*np.exp(1j*chi)*PsiM_s 
    return get_rho(E_state0_s)

def get_E0_p(eta, chi):
    '''Eritas's state of the form cos(eta)PhiP + e^(i*chi)*sin(eta)PhiM'''
    E_state0_s= np.cos(eta)*PhiP_s + np.sin(eta)*np.exp(1j*chi)*PhiM_s 
    return get_rho(E_state0_s)

def sample_E0_p():
    eta = np.deg2rad(30)
    chi_ls = np.linspace(0, np.pi/2, 6)
    W_ls = []
    Wp_ls = []
    for chi in chi_ls:
        rho = get_E0_p(eta, chi)
        W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho)
        Wp= min(Wp_t1, Wp_t2, Wp_t3)
        W_ls.append(W_min)
        Wp_ls.append(Wp)
    plt.figure(figsize=(10,10))
    chi_ls = np.rad2deg(chi_ls)
    plt.scatter(chi_ls, W_ls, label='W')
    plt.scatter(chi_ls, Wp_ls, label='Wp')
    plt.legend()
    plt.xlabel('$\chi$')
    plt.ylabel('Witness value')
    plt.title(f'$\eta = {np.round(np.rad2deg(eta), 3)}$')
    plt.savefig(f'E0_{eta}.pdf')

def get_E1(eta, chi):
    '''Eritas's state of the form 1/sqrt2 * cos(eta)*(PsiP + iPsiM) + e^(i*chi)*sin(eta)*(PhiP + iPhiM))'''
    E_state1_s= 1/np.sqrt(2) * (np.cos(eta)*(PsiP_s + 1j*PsiM_s) + np.sin(eta)*np.exp(1j*chi)*(PhiP_s + 1j*PhiM_s))
    return get_rho(E_state1_s)

## sample of different values of eta, chi for E0 and E1
def sample_E(exp_data=None):
    '''Samples E0 and E1 for various values of chi and eta and computes witness values'''
    eta_ls = np.linspace(0, np.pi/4, 6) # set of eta values to sample
    chi_ls = np.linspace(0, np.pi/2, 6) # set of chi values to sample
    # fixed angles to investigate for slice of plot 
    eta_fixed= np.pi/8
    chi_fixed= np.pi/2

    E0_W_ls = []
    E0_Wp_ls = []
    E1_W_ls = []
    E1_Wp_ls = []

    for chi in chi_ls:
        rho = get_E0(eta_fixed, chi)
        W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho)
        Wp= min(Wp_t1, Wp_t2, Wp_t3)
        E0_W_ls.append(W_min)
        E0_Wp_ls.append(Wp)

    for eta in eta_ls:
        rho = get_E1(eta, chi_fixed)
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

    def deg_formatter(x, pos):
        return str(np.round(np.rad2deg(x), 3))

    def sinsq2(x, a, b, c, d):
        return a*np.sin(b*x + c)**2 + d
    
    # plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].scatter(chi_ls, E0_W_ls, label='W')
    ax[0].scatter(chi_ls, E0_Wp_ls, label='W\'')

    popt_E0_W, pcov_E0_W = curve_fit(sinsq2, chi_ls, E0_W_ls)
    popt_E0_Wp, pcov_E0_Wp = curve_fit(sinsq2, chi_ls, E0_Wp_ls)

    chi_fit = np.linspace(0, np.pi/2, 100)
    ax[0].plot(chi_fit, sinsq2(chi_fit, *popt_E0_W), label='fit W')
    ax[0].plot(chi_fit, sinsq2(chi_fit, *popt_E0_Wp), label='fit W\'')

    ax[0].set_title('$\cos(\eta)|\Psi^+ \\rangle + e^{i\chi}\sin(\eta)|\Psi^- \\rangle, \\eta = %2g^\degree$'%np.rad2deg(eta_fixed))
    ax[0].set_xlabel('$\\chi (^\degree)$')
    ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(deg_formatter))
    ax[0].set_xticks(chi_ls)
    ax[0].set_ylabel('Witness value')
    ax[0].legend()

    if exp_data is not None:
        W_vals = exp_data['W_vals']
        W_errs = exp_data['W_errs']
        Wp_vals = exp_data['Wp_vals']
        Wp_errs = exp_data['Wp_errs']
        ax[0].errorbar(chi_ls, W_vals, yerr=W_errs, fmt='o', label='W exp')
        ax[0].errorbar(chi_ls, Wp_vals, yerr=Wp_errs, fmt='o', label='W\' exp')

    ax[1].scatter(eta_ls, E1_W_ls, label='W')
    ax[1].scatter(eta_ls, E1_Wp_ls, label='W\'')
    ax[1].set_xlabel('$\\eta (^\degree)$')
    ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(deg_formatter))
    ax[1].set_xticks(eta_ls)
    ax[1].set_ylabel('Witness value')

    popt_E1_W, pcov_E1_W = curve_fit(sinsq2, eta_ls, E1_W_ls)
    popt_E1_Wp, pcov_E1_Wp = curve_fit(sinsq2, eta_ls, E1_Wp_ls)

    eta_fit = np.linspace(0, np.pi/4, 100)
    ax[1].plot(eta_fit, sinsq2(eta_fit, *popt_E1_W), label='fit W')
    ax[1].plot(eta_fit, sinsq2(eta_fit, *popt_E1_Wp), label='fit W\'')
    ax[1].set_title('$\cos(\eta)(|\Psi^+ \\rangle + i|\Psi^-\\rangle) + e^{i\chi }\sin(\eta)(|\Phi^+ \\rangle + i|\Phi^-\\rangle), \\chi = %.2g^\degree$'%np.rad2deg(chi_fixed))
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Increase the 'top' value to add more space
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
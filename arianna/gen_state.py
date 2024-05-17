from qutip import *
from math import sqrt, cos, sin
import numpy as np
from scipy.optimize import minimize, approx_fprime

# generates a density matrix for the |E_0> states
# uses parameters eta and chi, these are tunable

# Basis Notes: 
H = basis(2,0) # ket 1
V = basis(2,1) # ket 2
D = 1/sqrt(2) * (H + V)
A = 1/sqrt(2) * (H - V)
R = 1/sqrt(2)*(H + 1j*V)
L = 1/sqrt(2)*(H - 1j*V)

# Bell state Psi = 1/sqrt(2) (|HV> +/- |VH>)
psi_p = 1/sqrt(2)*(tensor(H,V) + tensor(V,H))
psi_n = 1/sqrt(2)*(tensor(H,V) - tensor(V,H))

# Bell state Phi = 1/sqrt(2) (|HH> +/- |VV>)
phi_p = 1/sqrt(2)*(tensor(H,H) + tensor(V,V))
phi_n = 1/sqrt(2)*(tensor(H,H) - tensor(V,V))

def make_E0(eta, chi, type):
    """ Inputs: eta, chi - parameters to change the E0 function
        Outputs: E0 density matrix qobj 
    """
    #parameters eta, chi
    eta =  eta # radians
    chi = chi # radians
    if type == "psi":
        # generate E_0, the special missed state by {W}
        E0 = cos(eta)*psi_p + sin(eta)*np.exp(chi*1j)*psi_n
    elif type == "phi":
        # generate E_0, the special missed state by {W}
        E0 = cos(eta)*phi_p + sin(eta)*np.exp(chi*1j)*phi_n

    # generate the density matrix
    rho_E0 = ket2dm(E0)

    return rho_E0


def make_W_list(theta, Wtype):
    """ Inputs: real coefficients of phi 1...6 states a^2 + b^2 = 1
                Wtype - if Wtype is nonzero then we want to return the witness it corresponds to
        Outputs: 6 extremal entanglement witness with those coefficients
    """
    a = cos(np.radians(theta))
    b = sin(np.radians(theta))

    # List of six states possible from the given Bell states
    phi_1 = a*phi_p + b*phi_n
    phi_2 = a*psi_p + b*psi_n
    phi_3 = a*phi_p + b*psi_p
    phi_4 = a*phi_n + b*psi_n
    phi_5 = a*phi_p + 1j*b*psi_n
    phi_6 = a*phi_n + 1j*b*psi_p

    state_vecs = [phi_1,phi_2,phi_3,phi_4,phi_5,phi_6]


    WList = [partial_transpose(state_vecs[i]*state_vecs[i].dag(), [0,1]) for i in range(len(state_vecs))]

    if Wtype in range(1,7):
        return WList[Wtype-1]
    else:
        return WList

def make_WPrimes_list(alpha,beta,theta,Wtype):
    """ Inputs: coefficient parameters, radians, that determine states phi 1...9 
        Outputs: 9 extremal entanglement witness with those coefficients, W's
    """

    # List of 9 states possible
    phi_1 = cos(theta)*phi_p + np.exp(1j*alpha)*sin(theta)*phi_n
    phi_2 = cos(theta)*psi_p + sin(theta)*psi_n
    phi_3 = 1/sqrt(2)*(cos(theta)*tensor(H,H) + np.exp(1j*(beta-alpha))*sin(theta)*tensor(H,V) + np.exp(1j*alpha)*sin(theta)*tensor(V,H)+np.exp(1j*beta)*cos(theta)*tensor(V,V))
    phi_4 = cos(theta)*phi_p + np.exp(1j*alpha)*sin(theta)*psi_p
    phi_5 = cos(theta)*phi_n + np.exp(1j*alpha)*sin(theta)*psi_n
    phi_6 = cos(theta)*cos(alpha)*tensor(H,H) + 1j*cos(theta)*sin(alpha)*tensor(H,V) + 1j*sin(theta)*sin(beta)*tensor(V,H)+sin(theta)*cos(beta)*tensor(V,V)
    phi_7 = cos(theta)*phi_p + np.exp(1j*alpha)*sin(theta)*psi_n
    phi_8 = cos(theta)*phi_n + np.exp(1j*alpha)*sin(theta)*psi_p
    phi_9 = cos(theta)*cos(alpha)*tensor(H,H) + cos(theta)*sin(alpha)*tensor(H,V) + sin(theta)*sin(beta)*tensor(V,H)+sin(theta)*cos(beta)*tensor(V,V)

    state_vecs = [phi_1,phi_2,phi_3,phi_4,phi_5,phi_6, phi_7, phi_8, phi_9]


    WPrimeList = [partial_transpose(state_vecs[i]*state_vecs[i].dag(), [0,1]) for i in range(len(state_vecs))]
    
    if Wtype in range(1,10):
        return WPrimeList[Wtype-1]
    else:
        return WPrimeList




"""
1: eta = 0...2pi in 3, exclude 0 chi = 0...2pi in 3
[1.2523902708318734e-17, 1.2523902708318734e-17, 0.2499999999999999, 0.2499999999999999, 4.163336342344337e-17, 4.163336342344337e-17, 1.6208531202556912e-19, 1.2476119542561794e-17, 0.0, 0.0, 1.086630785351872e-14, -2.7755575615628914e-17, 1.7208456881689926e-15, 0.0, 0.0]
[[0], [0], [3.141592653589793], [3.141592653589793], [0], [0], [1.2831565104473828, 1.214446753109363], [1.5707963267948966, 6.283185307179586], [0.8039161499292756, 2.789239807767016, 2.9899145512522614], [1.2541434294298106, 5.7272262085133825], [0.21053778045466154, 3.160498949563602], [1.5707963267948966, 6.283185307179586, 6.283185307179586], [1.5077286081296557, 0.8424579037545278], [1.1831967873620906, 2.99222375573456], [0, 0, 0]]
[1.5707963267948966, 6.283185307179586, 6.283185307179586]

W'6
Should be for state 2, check different eta and chi
"""
eta = np.radians(2*np.pi)
chi = np.radians(2*np.pi)

test_1 = (cos(eta)*tensor(H,L) + np.exp(1j*chi)*sin(eta)*tensor(L,H))/((1 + 0.5*cos(chi)*sin(2*eta))**(0.5))
rho_1 = test_1*test_1.dag()

test_2 = (cos(eta)*tensor(V,R) + np.exp(1j*chi)*sin(eta)*tensor(R,V))/((1 + 0.5*cos(chi)*sin(2*eta))**(0.5))
rho_2 = test_2*test_2.dag()

test_3 = (cos(eta)*tensor(V,L) + np.exp(1j*chi)*sin(eta)*tensor(V,L))/((1 + 0.5*cos(chi)*sin(2*eta))**(0.5))
rho_3 = test_3*test_3.dag()

test_4 = (cos(eta)*tensor(H,R) + np.exp(1j*chi)*sin(eta)*tensor(R,H))/((1 + 0.5*cos(chi)*sin(2*eta))**(0.5))
rho_4 = test_4*test_4.dag()

theta = 1.5707963267948966
alpha = 6.283185307179586
beta = 6.283185307179586

Wp6 = make_WPrimes_list(alpha,beta,theta,6)
print('------ 1 -------')
print(test_1)
print(rho_1)

print((Wp6*rho_1).tr())

print('------ 2 -------')
print(test_2)
print(rho_2)

print((Wp6*rho_2).tr())

print('------ 3 -------')
print(test_3)
print(rho_3)

print((Wp6*rho_3).tr())


print('------ 3 -------')
print(test_3)
print(rho_3)

print((Wp6*rho_3).tr())

print('------ 4 -------')
print(test_4)
print(rho_4)

print((Wp6*rho_4).tr())
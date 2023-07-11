''' qo_tools.py

This module has some helpful functions and constants for playing around with quantum optics.

'''

import numpy as np
from numpy import sin, cos, tan, arcsin, arctan, arctan2, sqrt, exp, pi, log, angle, rad2deg, deg2rad

# constants

I = 1j

def expi(x):
    return exp(I*x)

# one qubit states

def ket(x):
    return np.array(x).reshape(-1,1)

def bra(x):
    return np.array(x).reshape(1,-1)

def adj(x):
    return x.conj().T

# one qubit states

H = ket([1,0])
V = ket([0,1])
D = ket([1,1])/np.sqrt(2)
A = ket([1,-1])/np.sqrt(2)
R = ket([1,1j])/np.sqrt(2)
L = ket([1,-1j])/np.sqrt(2)

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

PHI_PLUS = ket([1,0,0,1])/np.sqrt(2)
PHI_MINUS = ket([1,0,0,-1])/np.sqrt(2)
PSI_PLUS = ket([0,1,1,0])/np.sqrt(2)
PSI_MINUS = ket([0,1,-1,0])/np.sqrt(2)

# imperfect HWP

def rotmat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def imperfect_HWP(theta, gamma):
    return rotmat(theta) @ np.array([[1,0],[0,-np.exp(1j*gamma)]]) @ rotmat(-theta)

def hwp(theta):
    return rotmat(theta) @ np.array([[1,0],[0,-1]]) @ rotmat(-theta)

def qwp(theta):
    return rotmat(theta) @ np.array([[1,0],[0,1j]]) @ rotmat(-theta)

# general state

def gen_prod(alpha, beta, phi):
    return ket([
        np.cos(alpha)*np.cos(beta),
        np.cos(alpha)*np.sin(beta),
        np.sin(alpha)*np.sin(beta)*np.exp(1j*phi),
        -np.sin(alpha)*np.cos(beta)*np.exp(1j*phi)])

def dot(x,y):
    return (adj(x) @ y).item()

def proj(x,y):
    return np.abs(dot(x,y))**2


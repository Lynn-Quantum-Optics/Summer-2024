import numpy as np
from rho_methods import *
from sample_rho import *

# HR + RH
HR = HH + 1j*VH
RH = HH + 1j*HV
state = HR + RH
# normalize
state = state/np.linalg.norm(state)
state_mat = get_rho(state)

print(compute_witnesses(state_mat, return_all=True))

# let's compute explicitly what Wp4 is
from sympy import *
import sympy as sp
# alpha and theta are real
alpha, theta = sp.symbols('alpha theta', real=True)
PHI_P = sp.Matrix([[1/sp.sqrt(2), 0, 0, 1/sp.sqrt(2)]])
PHI_P = PHI_P.reshape(4,1)
PHI_M = sp.Matrix([[1/sp.sqrt(2), 0, 0, -1/sp.sqrt(2)]])
PHI_M = PHI_M.reshape(4,1)
PSI_P = sp.Matrix([[0, 1/sp.sqrt(2),  1/sp.sqrt(2), 0]])
PSI_P = PSI_P.reshape(4,1)
PSI_M = sp.Matrix([[0, 1/sp.sqrt(2),  -1/sp.sqrt(2), 0]])
PSI_M = PSI_M.reshape(4,1)
# column vectors
HH_sp = sp.Matrix([[1, 0, 0, 0]]).reshape(4,1)
HV_sp = sp.Matrix([[0, 1, 0, 0]]).reshape(4,1)
VH_sp = sp.Matrix([[0, 0, 1, 0]]).reshape(4,1)
VV_sp = sp.Matrix([[0, 0, 0, 1]]).reshape(4,1)

print('state in question')
state = HH_sp + 1j*VH_sp + HH_sp + 1j*HV_sp
state = state/state.norm()
sp.print_latex(sp.simplify(state))  
state_mat = state @ adjoint(state)
sp.print_latex(sp.simplify(state_mat))

print('New W proposal')
# phi4_p = sp.cos(theta)*PHI_P + sp.exp(1j*alpha)*sp.sin(theta)*PSI_P
# arianna's proposal: HH + RR
phi4_p = sp.cos(theta) * (HH_sp) + sp.exp(1j*alpha) * sp.sin(theta) * (HH_sp + sp.I*HV_sp + sp.I*VH_sp - VV_sp)
sp.print_latex(sp.simplify(phi4_p))

print('----------')
print('W4 state density matrix')

# compute density matrix
phi4_mat = phi4_p * adjoint(phi4_p)
sp.print_latex(sp.simplify(phi4_mat))

# take partial transpose
print('----------')
print('W4 state density matrix partial transpose')

def partial_transpose_sympy(rho, subsys='B'):
    ''' Helper function to compute the partial transpose of a density matrix. Useful for the Peres-Horodecki criterion, which states that if the partial transpose of a density matrix has at least one negative eigenvalue, then the state is entangled.
    Params:
        rho: density matrix
        subsys: which subsystem to compute partial transpose wrt, i.e. 'A' or 'B'
    '''
    # decompose rho into blocks
    b1 = rho[:2, :2]
    b2 = rho[:2, 2:]
    b3 = rho[2:, :2]
    b4 = rho[2:, 2:]

    PT = sp.Matrix(sp.BlockMatrix([[b1.T, b2.T], [b3.T, b4.T]]))

    if subsys=='B':
        return PT
    elif subsys=='A':
        return PT.T

phi4_pt = partial_transpose_sympy(phi4_mat)
sp.print_latex(sp.simplify(phi4_pt))

# compute Wp4
print('----------')
print('Wp4')
Wp4 = sp.simplify(sp.trace(phi4_pt @ state_mat))
sp.print_latex(Wp4)

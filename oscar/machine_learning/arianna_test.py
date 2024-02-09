# import numpy as np
# # from rho_methods import *
# from sample_rho import *

# # HR + RH
# HR = HH + sp.I*VH
# RH = HH + sp.I*HV
# state = HR + RH
# # normalize
# state = state/np.linalg.norm(state)
# state_mat = get_rho(state)

# print(compute_witnesses(state_mat, return_all=True))

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

print('HH')
print(HH_sp)

RR_sp = HH_sp + sp.I*HV_sp + sp.I*VH_sp - VV_sp
RR_sp = RR_sp/RR_sp.norm()

RH_sp = HH_sp + sp.I*VH_sp
RH_sp = RH_sp/RH_sp.norm()
HR_sp = HH_sp + sp.I*HV_sp
HR_sp = HR_sp/HR_sp.norm()

# print('state in question')
# state = HH_sp + sp.I*VH_sp + HH_sp + sp.I*HV_sp
state = sp.cos(theta)*HH_sp + sp.sin(theta)*sp.exp(sp.I*alpha)*(RH_sp)
# a and b are real numbers
# a, b = sp.symbols('a b', complex=True)
# state = a*HR_sp + b*RH_sp
state = state/state.norm()
sp.print_latex(sp.simplify(state))  
state_mat = state * state.T.conjugate() 
sp.print_latex(sp.simplify(state_mat))

print('New W proposal')
# phi4_p = sp.cos(theta)*PHI_P + sp.exp(sp.I*alpha)*sp.sin(theta)*PSI_P
# arianna's proposal: HH + RR
# phi4_p = sp.cos(theta) * (HH_sp) + sp.exp(sp.I*alpha) * sp.sin(theta) * (HH_sp + sp.I*HV_sp + sp.I*VH_sp - VV_sp)
phi4_p = HH_sp - RR_sp
sp.print_latex(sp.simplify(phi4_p))

print('----------')
print('W4 state density matrix')

# compute density matrix
phi4_mat = phi4_p * phi4_p.T.conjugate()
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

print('********')
sp.print_latex(phi4_pt)
print('**')
sp.print_latex(state_mat)
print('********')

sp.print_latex(sp.simplify(phi4_pt * state_mat))

print('---------')
Wp4 = sp.simplify(sp.trace(phi4_pt * state_mat))
sp.pprint(sp.simplify(sp.simplify(sp.re(Wp4))))

# solution = solve(Lt(Wp4, 0), (alpha, theta))
# print(solution)

# minimize this
from oscars_toolbox.trabbit import trabbit
import numpy as np

# convert to numpy by defining it as a function of alpha and theta
def f(vals):
    alpha_val, theta_val = vals
    expr_val = Wp4.subs({alpha: alpha_val, theta: theta_val}).evalf()
    # Extract the real part and convert to a NumPy float for compatibility with other NumPy operations
    real_val = float(sp.re(expr_val))

    return real_val

def random_generator():
    alpha = np.random.uniform(0, 2*np.pi)
    theta = np.random.uniform(0, 2*np.pi)
    return [alpha,theta]

x_best, loss_best = trabbit(loss_func = f, random_gen=random_generator, alpha=0.7, num=1000, tol=1e-6)
print(x_best, loss_best)

# # try a grid search for what the minimum is
# alpha_vals = np.linspace(-np.pi, np.pi, 100)
# theta_vals = np.linspace(-np.pi/2, np.pi/2, 100)

# # see what the function looks like
# # create meshgrid
# alpha_grid, theta_grid = np.meshgrid(alpha_vals, theta_vals)
# f_vals = np.zeros(alpha_grid.shape)
# for i in range(alpha_grid.shape[0]):
#     for j in range(alpha_grid.shape[1]):
#         f_vals[i,j] = f([alpha_grid[i,j], theta_grid[i,j]])

# # find minimum
# min_idx = np.unravel_index(np.argmin(f_vals, axis=None), f_vals.shape)
# print('Minimum value:', f_vals[min_idx])



# # print('-------')


# from scipy.optimize import minimize
# print(minimize(f, [0.7, 0.7], bounds=[(0, 2*np.pi), (0, 2*np.pi)])['fun'])


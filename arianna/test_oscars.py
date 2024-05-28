from qutip import *
from math import sqrt, cos, sin
import numpy as np
# from gen_state import make_W, make_E0, make_WPrimes
from rho_methods_ONLYGOOD import compute_witnesses, get_rho
import matplotlib.pyplot as plt

state = np.array([[1],[0],[0],[0]])
rho = get_rho(state) 
print(rho)
wit_expectation = compute_witnesses(rho)
print("with matrices", wit_expectation)

# state2 = np.array([[1],[0],[0],[0]])
state2 = tensor(basis(2,0), basis(2,0))
# rho2 = get_rho(state2) 
rho2 = ket2dm(state2).full()
print(rho2)
# rho = np.matrix([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])
wit_expectation2 = compute_witnesses(rho2)
print("with qutip", wit_expectation2)


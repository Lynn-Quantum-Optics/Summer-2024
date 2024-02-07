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
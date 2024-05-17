
import numpy as np
import scipy.linalg as la
import uncertainties.unumpy as unp
from full_tomo import reconstruct_rho

# hopefully this
# (rho_real, rho_imag, stokes, un_proj) = np.load('rho_out.npy', allow_pickle=True)

# otherwise this
bases = list('HVDARL')
from lab_framework import Manager
df = Manager.load_data('tomography_data.csv')

projs = np.zeros((6,6), dtype=object)
for i in range(len(df)):
    a, b = df['note'][i]
    a, b = bases.index(a), bases.index(b)
    projs[a,b] = df['C4'][i]

# normalize groups of orthonormal measurements to get projections
for i in range(0,6,2):
    for j in range(0,6,2):
        total_rate = np.sum(projs[i:i+2, j:j+2])
        projs[i:i+2, j:j+2] /= total_rate

rho_real, rho_imag, stokes = reconstruct_rho(projs)

# then do the analysis
rho = unp.nominal_values(rho_real) + 1j*unp.nominal_values(rho_imag)
phi_plus = np.array([1,0,0,1]).reshape(4,1)/np.sqrt(2)
phi_plus = phi_plus @ phi_plus.T

pur = np.trace(rho @ rho)
fid = np.real(np.trace(la.sqrtm((la.sqrtm(phi_plus)@rho@la.sqrtm(phi_plus))))**2)

# print
print(f'Purity = {pur}')
print(f'Fidelity = {fid}')
# print 
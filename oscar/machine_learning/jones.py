# file for sample jones matrix computations
import numpy as np

## sample bell states##
PhiP = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
rho_PhiP= PhiP @ PhiP.reshape((1,4))
print(rho_PhiP, 'PhiP')

PhiM = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).reshape((4,1))
rho_PhiM= PhiM @ PhiM.reshape((1,4))
print(rho_PhiM, 'PhiM')

PsiP = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0]).reshape((4,1))
rho_PsiP= PsiP @ PsiP.reshape((1,4))
print(rho_PsiP, 'PsiP')

PsiM = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0]).reshape((4,1))
rho_PsiM= PsiM @ PsiM.reshape((1,4))
print(rho_PsiM, 'PsiM')

## jones matrices ##
def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): return R(alpha) @ np.matrix(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def QP(phi): return np.matrix(np.diag([1, np.e**(phi*1j)]))

## test sample values ##
# angles are: H1, H2, Q1, QP1
angle_dict = {'PhiP':[np.pi/4, 0,0, 0, 0], 'PhiM':[np.pi/4, 0, 0, 0, np.pi], 'PsiP':[np.pi/4, 0, np.pi/4,np.pi/4,  np.pi], 'PsiM':[np.pi/4, 0, np.pi/4,np.pi/4, 0]}
desired_state = angle_dict['PsiP']

## compute components ##
H1 = H(desired_state[0])
H2 = H(desired_state[1])
Q1 = Q(desired_state[2])
Q2 = Q(desired_state[3])
QP1 = QP(desired_state[4])

## compute density matrix ##
M = np.kron(H1, H2 @ Q2 @  Q1 @ QP1 @ H1) @ rho_PhiP
rho = np.round(M @ M.H,2).real

print(rho)
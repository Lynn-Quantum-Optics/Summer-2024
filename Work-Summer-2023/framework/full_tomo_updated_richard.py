''' full_tomo.py

This file contains functions for executing a full tomographic measurement of any state that one configures. The function that completes the tomography itself is get_rho.

authors: Alec Roberson (aroberson@hmc.edu), Oscar Scholin (osscholin@hmc.edu)
'''
from typing import Tuple
import numpy as np
from lab_framework import Manager
import sys

sys.path.insert(0, '../oscar/machine_learning')
from rho_methods import get_fidelity, get_purity

def reconstruct_rho(all_projs:np.ndarray, all_proj_uncs:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ''' Takes in all 36 projections and reconstructs the density matrix. Based on Beili Hu's thesis.
    
    Adapted from Oscar's version of the same function.

    Parameters
    ----------
    all_projs : numpy.ndarray of shape (6,6)
        The 36 projections into the powerset of H, V, D, A, R, and L.
    all_proj_uncs : numpy.ndarray of shape (6,6)
        Uncertainty values for the 36 projections into the powerset of H, V, D, A, R, and L.
    
    Returns
    -------
    numpy.ndarray of shape (4,4)
        The reconstructed density matrix.
    numpy.ndarray of shape (4,4)
        Uncertainties in the reconstructed density matrix.
    
    '''
    
    # +++ oscar's code

    # unpack the projections
    HH, HV, HD, HA, HR, HL = all_projs[0]
    VH, VV, VD, VA, VR, VL = all_projs[1]
    DH, DV, DD, DA, DR, DL = all_projs[2]
    AH, AV, AD, AA, AR, AL = all_projs[3]
    RH, RV, RD, RA, RR, RL = all_projs[4]
    LH, LV, LD, LA, LR, LL = all_projs[5]

    # build the stokes's parameters
    S = np.zeros((4,4))
    S[0,0]=1
    S[0,1] = DD - DA + AD - AA
    S[0,2] = RR + LR - RL - LL
    S[0,3] = HH - HV + VH - VV
    S[1,0] = DD + DA - AD - AA
    S[1,1] = DD - DA - AD + AA
    S[1,2] = DR - DL - AR + AL
    S[1,3] = DH - DV - AH + AV
    S[2,0] = RR - LR + RL - LL
    S[2,1] = RD - RA - LD + LA
    S[2,2] = RR - RL - LR + LL
    S[2,3] = RH - RV - LH + LV
    S[3,0] = HH + HV - VH - VV
    S[3,1] = HD - HA - VD + VA
    S[3,2] = HR - HL - VR + VL
    S[3,3] = HH - HV - VH + VV

    # define pauli matrices
    I = np.eye(2)
    X = np.matrix([[0, 1], [1, 0]])
    Y = np.matrix([[0, -1j], [1j, 0]])
    Z = np.matrix([[1, 0], [0, -1]])
    P = [I, X, Y, Z]

    # compute rho
    rho = np.zeros((4,4), dtype=complex)
    for i1 in range(4):
        for i2 in range(4):
            rho+= S[i1,i2]*np.kron(P[i1],P[i2])

    # scale by 4 to get the correct density matrix
    rho = rho/4

    # +++ alec's code

    # unpack the projections
    HHu, HVu, HDu, HAu, HRu, HLu = all_proj_uncs[0]
    VHu, VVu, VDu, VAu, VRu, VLu = all_proj_uncs[1]
    DHu, DVu, DDu, DAu, DRu, DLu = all_proj_uncs[2]
    AHu, AVu, ADu, AAu, ARu, ALu = all_proj_uncs[3]
    RHu, RVu, RDu, RAu, RRu, RLu = all_proj_uncs[4]
    LHu, LVu, LDu, LAu, LRu, LLu = all_proj_uncs[5]

    # build the stokes's parameter uncertainties
    Su = np.zeros((4,4))
    Su[0,0] = 0
    Su[0,1] = np.sqrt(DDu**2 + DAu**2 + ADu**2 + AAu**2)
    Su[0,2] = np.sqrt(RRu**2 + LRu**2 + RLu**2 + LLu**2)
    Su[0,3] = np.sqrt(HHu**2 + HVu**2 + VHu**2 + VVu**2)
    Su[1,0] = np.sqrt(DDu**2 + DAu**2 + ADu**2 + AAu**2)
    Su[1,1] = np.sqrt(DDu**2 + DAu**2 + ADu**2 + AAu**2)
    Su[1,2] = np.sqrt(DRu**2 + DLu**2 + ARu**2 + ALu**2)
    Su[1,3] = np.sqrt(DHu**2 + DVu**2 + AHu**2 + AVu**2)
    Su[2,0] = np.sqrt(RRu**2 + LRu**2 + RLu**2 + LLu**2)
    Su[2,1] = np.sqrt(RDu**2 + RAu**2 + LDu**2 + LAu**2)
    Su[2,2] = np.sqrt(RRu**2 + RLu**2 + LRu**2 + LLu**2)
    Su[2,3] = np.sqrt(RHu**2 + RVu**2 + LHu**2 + LVu**2)
    Su[3,0] = np.sqrt(HHu**2 + HVu**2 + VHu**2 + VVu**2)
    Su[3,1] = np.sqrt(HDu**2 + HAu**2 + VDu**2 + VAu**2)
    Su[3,2] = np.sqrt(HRu**2 + HLu**2 + VRu**2 + VLu**2)
    Su[3,3] = np.sqrt(HHu**2 + HVu**2 + VHu**2 + VVu**2)

    # compute rho uncertainty
    rho_unc = np.zeros((4,4), dtype=complex)
    for i1 in range(4):
        for i2 in range(4):
            rho_unc += np.square(Su[i1,i2]*np.kron(P[i1],P[i2]))
    
    # take the sqrt and divide by 4 to get the correct uncertainty
    rho_unc = np.sqrt(rho_unc)/4

    # return density matrix with uncertainties and the Stokes params unc for witness unc
    return rho, rho_unc, Su

def get_projections(m:Manager, samp:Tuple[int,float]) -> Tuple[np.ndarray, np.ndarray]:
    ''' Get all 36 projective measurments (and uncertainties) in the powerset of H, V, D, A, R, and L.

    Parameters
    ----------
    m : Manager
        The manager object to use for taking data.
    samp : Tuple[int,float]
        The sample to use for taking data. The first element is the sample number, and the second is the exposure time.
    
    Returns
    -------
    numpy.ndarray of shape (6,6)
        The 36 projections into the powerset of H, V, D, A, R, and L.
    numpy.ndarray of shape (6,6)
        Uncertainties for each of the 36 projections.

    '''
    # setup output
    proj = []
    proj_unc = []

    # loop through bases
    bases = 'HVDARL'
    for a in bases:
        row = []
        row_unc = []
        for b in bases:
            # set the measurement basis
            m.meas_basis(a+b)
            # take the sample
            x = m.take_data(*samp, 'C4', note=f'{a+b}')
            row.append(x.nominal_value)
            row_unc.append(x.std_dev)
        proj.append(row)
        proj_unc.append(row_unc)

    # make projs into arrays
    proj = np.array(proj)
    proj_unc = np.array(proj_unc)

    # save unnormalized proj
    un_proj = proj.copy()
    un_proj_unc = proj_unc.copy()

    # normalize groups of orthonormal measurements to get projections
    for i in range(0,6,2):
        for j in range(0,6,2):
            total_rate = np.sum(proj[i:i+2, j:j+2])
            proj[i:i+2, j:j+2] /= total_rate
            proj_unc[i:i+2, j:j+2] /= total_rate
    
    # return the projections and uncertainties
    return proj, proj_unc, un_proj, un_proj_unc

# methods for finding the density matrix of a given state
def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho(alpha, beta):

    H = ket([1,0])
    V = ket([0,1])

    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)

    psi = np.cos(alpha)*PSI_PLUS + np.exp(1j*beta)*np.sin(alpha)*PSI_MINUS

    rho = psi @ psi.conj().T

    return rho

def get_rho(m:Manager, samp:Tuple[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    ''' Get the density matrix and uncertainty for the state that is currently configured.

    Parameters
    ----------
    m : Manager
        The manager object.
    samp : Tuple[int,float]
        Sampling parameters for the trial.
    
    Returns
    -------
    numpy.ndarray of shape (4,4)
        The density matrix for the state.
    numpy.ndarray of shape (4,4)
        The uncertainty for the density matrix.
    '''
    proj, proj_unc, un_proj, un_proj_unc = get_projections(m, samp)
    rho, rho_unc, Su = reconstruct_rho(proj, proj_unc)
    return rho, rho_unc, Su, un_proj, un_proj_unc

if __name__ == '__main__':
    SAMP = (5, 1)

    # open manager
    m = Manager()
    
    # load configured state
    # m.make_state('phi_plus')
    m.C_QP.goto(-31.54505226)
    m.C_UV_HWP.goto(40.95054)
    m.B_C_HWP.goto(45)
    m.C_PCC.goto(4.005)

    # get the density matrix
    rho, unc, Su, un_proj, un_proj_unc = get_rho(m, SAMP)

    actual_rho = get_theo_rho(np.pi/4, .001)

    # print results
    print('RHO\n---')
    print(rho)
    print('UNC\n---')
    print(unc)
    print('actual rho\n ---')
    print(actual_rho)

    fidelity = get_fidelity(rho, actual_rho)
    print('fidelity', fidelity)
    purity = get_purity(rho)
    print('purity', purity)

    state = (np.pi/4, .001)
    angles = [328.45494774, 40.95054, 45]
    state_n = (np.rad2deg(np.pi/4), np.rad2deg(.001))

    # save results
    with open(f'int_state_sweep/rho_{state_n}.npy', 'wb') as f:
        np.save(f, (rho, unc, Su, un_proj, un_proj_unc, state, angles, fidelity, purity))
        
    tomo_df = m.output_data(f'int_state_sweep/tomo_data_{state}.csv')

    # close out
    # m.close_output()
    m.shutdown()
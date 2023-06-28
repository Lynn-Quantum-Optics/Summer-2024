''' full tomography script

authors: Alec Roberson (aroberson@hmc.edu), Oscar Scholin (osscholin@hmc.edu)
'''
from typing import Tuple
import numpy as np
from core import Manager


def reconstruct_rho(all_projs:np.ndarray, all_proj_uncs:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ''' Takes in all 36 projections and reconstructs the density matrix. Based on Beili Hu's thesis.
    
    This function written by Oscar.

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
    Su[0,0]=1
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

    # return density matrix with uncertainties
    return rho, rho_unc

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
            x, xu = m.take_data(*samp, 'C4', note=f'{a+b}')
            row.append(x)
            row_unc.append(xu)
        proj.append(row)
        proj_unc.append(row_unc)

    # make projs into arrays
    proj = np.array(proj)
    proj_unc = np.array(proj_unc)

    # normalize groups of orthonormal measurements to get projections
    for i in range(0,6,2):
        for j in range(0,6,2):
            total_rate = np.sum(proj[i:i+2, j:j+2])
            proj[i:i+2, j:j+2] /= total_rate
            proj_unc[i:i+2, j:j+2] /= total_rate
    
    # return the projections and uncertainties
    return proj, proj_unc

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
    return reconstruct_rho(*get_projections(m, samp))


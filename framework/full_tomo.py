''' full_tomo.py

This file contains functions for executing a full tomographic measurement of any state that one configures. The function that completes the tomography itself is get_rho.

authors: Alec Roberson (aroberson@hmc.edu), Oscar Scholin (osscholin@hmc.edu)
'''
from typing import Tuple
import numpy as np
from lab_framework import Manager

def reconstruct_rho(all_projs:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    S = np.zeros((4,4), dtype=object)
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
    rho_real = np.zeros((4,4), dtype=object)
    rho_imag = np.zeros((4,4), dtype=object)
    for i1 in range(4):
        for i2 in range(4):
            rho_real += S[i1,i2]*np.array(np.real(np.kron(P[i1],P[i2])), dtype=object)
            rho_imag += S[i1,i2]*np.array(np.imag(np.kron(P[i1],P[i2])), dtype=object)

    # scale by 4 to get the correct density matrix
    rho_real = rho_real/4
    rho_imag = rho_imag/4

    return rho_real, rho_imag, S

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

    # loop through bases
    bases = 'HVDARL'
    for a in bases:
        row = []
        for b in bases:
            # set the measurement basis
            m.meas_basis(a+b)
            # take the sample
            x = m.take_data(*samp, 'C4', note=f'{a+b}')
            row.append(x)
        proj.append(row)

    # make projs into arrays
    proj = np.array(proj)

    # save unnormalized proj
    un_proj = proj.copy()

    # normalize groups of orthonormal measurements to get projections
    for i in range(0,6,2):
        for j in range(0,6,2):
            total_rate = np.sum(proj[i:i+2, j:j+2])
            proj[i:i+2, j:j+2] /= total_rate
    
    # return the projections and uncertainties
    return proj, un_proj

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
    proj, un_proj = get_projections(m, samp)
    m.output_data('tomography_data_05222025_HDVA.csv')
    rho_real, rho_imag, stokes = reconstruct_rho(proj)
    return rho_real, rho_imag, stokes, un_proj

if __name__ == '__main__':
    SAMP = (5, 3)

    # open manager
    m = Manager()
    
    # load configured state
    m.make_state('phi_plus')

    # for hdva state
    m.B_C_HWP.goto(67.5)
    m.C_QP.goto(-21.288)
    m.C_UV_HWP.goto(-66.02461563913445)
    m.B_C_QWP.goto(45)

    # get the density matrix
    rho_real, rho_imag, stokes, un_proj = get_rho(m, SAMP)
    m.shutdown()

    # save results
    with open('rho_out.npy', 'wb') as f:
        np.save(f, (rho_real, rho_imag, stokes, un_proj))
    
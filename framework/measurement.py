''' state_measurement.py

This file contains functions that allow for us to measure the "parameters" of the quantum state being produced by our setup. The "parameters" in question are alpha, beta, and phi, where any state produced by our setup can be represented as

|psi> = cos(alpha)*cos(beta)*|HH>
    + cos(alpha)*sin(beta)*|HV>
    + sin(alpha)*sin(beta)*e^(i*phi)*|VH>
    - sin(alpha)*cos(beta)*e^(i*phi)*|VV>

In terms of those parameters. See the confluence page for more information about this script and an explaination of the maths.
'''

from typing import Tuple
import numpy as np

def calc_alpha(HD_HA_VD_VA:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], HD_HA_VD_VA_unc:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    ''' Calculate the alpha parameter of a state.

    Parameters
    ----------
    HD_HA_VD_VA : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple of count rate measurments in the HD, HA, VD, and VA bases.
    HD_HA_VD_VA_unc : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple of uncertainties in the count rate measurments in the HD, HA, VD, and VA bases.
    
    Returns
    -------
    np.ndarray
        The alpha parameter of the state.
    np.ndarray
        Uncertainty in the alpha parameter.
    '''
    # unpack the data
    HD, HA, VD, VA = HD_HA_VD_VA
    HDu, HAu, VDu, VAu = HD_HA_VD_VA_unc
    
    # get total count rates and convert everything to expectation values
    T = HD + HA + VD + VA
    HD, HA, VD, VA = HD/T, HA/T, VD/T, VA/T
    HDu, HAu, VDu, VAu = HDu/T, HAu/T, VDu/T, VAu/T

    # compute alpha
    alpha = np.arctan(np.sqrt((VD + VA)/(HD + HA)))
    # compute alpha uncertainty
    unc = 1/(2) * np.sqrt((HDu**2 + HAu**2) * (VD+VA)/(HD+HA) + (VDu**2 + VAu**2) * (HD+HA)/(VD+VA))

    return np.rad2deg(alpha), np.rad2deg(unc)

def calc_beta(HD_HA_VD_VA:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], HD_HA_VD_VA_unc:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    ''' Calculate the beta parameter of a state.

    Parameters
    ----------
    HD_HA_VD_VA : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple of count rate measurments in the HD, HA, VD, and VA bases.
    HD_HA_VD_VA_unc : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple of uncertainties in the count rate measurments in the HD, HA, VD, and VA bases.
    
    Returns
    -------
    np.ndarray
        The beta parameter of the state.
    np.ndarray
        Uncertainty in the beta parameter.
    '''
    # unpack the data
    HD, HA, VD, VA = HD_HA_VD_VA
    HDu, HAu, VDu, VAu = HD_HA_VD_VA_unc
    
    # get total count rates and convert everything to expectation values
    T = HD + HA + VD + VA
    HD, HA, VD, VA = HD/T, HA/T, VD/T, VA/T
    HDu, HAu, VDu, VAu = HDu/T, HAu/T, VDu/T, VAu/T

    # compute beta
    u = HD - HA - VD + VA
    beta = np.arcsin(u)
    # compute beta uncertainty
    unc = 1/np.sqrt(1-u**2) * np.sqrt(HDu**2 + HAu**2 + VDu**2 + VAu**2)

    return np.rad2deg(beta), np.rad2deg(unc)

def calc_phi(DL_DR_RR_RL:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], DL_DR_RR_RL_unc:Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    ''' Measure phi parameter of the state.

    Parameters
    ----------
    DL_DR_RR_RL : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple of count rate measurments in the DL, DR, RR, and RL bases.
    DL_DR_RR_RL_unc : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple of uncertainties in the count rate measurments in the DL, DR, RR, and RL bases.
    
    Returns
    -------
    np.ndarray
        The phi parameter of the state.
    np.ndarray
        Uncertainty in the phi parameter.
    '''
    # unpack the data
    DL, DR, RR, RL = DL_DR_RR_RL
    DLu, DRu, RRu, RLu = DL_DR_RR_RL_unc

    # get total count rates and convert everything to expectation values
    T = DL + DR + RR + RL
    DL, DR, RR, RL = DL/T, DR/T, RR/T, RL/T
    DLu, DRu, RRu, RLu = DLu/T, DRu/T, RRu/T, RLu/T

    # calculate phi
    phi = np.arctan2(DL-DR, RR-RL)
    # calculate uncertainty in phi
    u = (DL-DR)/(RR-RL)
    unc = 1/(1+u**2) * 1/np.abs(RR-RL) * np.sqrt(DLu**2 + DRu**2 + u**2 * (RRu**2 + RLu**2))

    return np.rad2deg(phi), np.rad2deg(unc)

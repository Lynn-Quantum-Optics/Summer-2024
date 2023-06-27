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
import pandas as pd
from core import Manager


def meas_HV(m:Manager, samp:Tuple[int,float]):
    ''' Takes data in (HH, HV, VH, VV) basis.
    
    Parameters
    ----------
    m : Manager
        The manager object that is controlling the experiment.
    samp : Tuple[int,float]
        The sample parameters for each measurement.
    
    Returns
    -------
    list
        The count rates for HH, HV, VH, and VV coincidences, in order.
    list
        The uncertainties for HH, HV, VH, and VV coincidences, in order.
    '''
    m.meas_basis("HH")
    HH, HH_unc = m.take_data(*samp, "C4")
    m.meas_basis("HV")
    HV, HV_unc = m.take_data(*samp, "C4")
    m.meas_basis("VH")
    VH, VH_unc = m.take_data(*samp, "C4")
    m.meas_basis("VV")
    VV, VV_unc = m.take_data(*samp, "C4")
    return [HH, HV, VH, VV], [HH_unc, HV_unc, VH_unc, VV_unc]

def meas_DA_RL(m:Manager, samp:Tuple[int,float]):
    ''' Takes data in (DR, DL, AR, AL) basis.
    
    Parameters
    ----------
    m : Manager
        The manager object that is controlling the experiment.
    samp : Tuple[int,float]
        The sample parameters for each measurement.
    
    Returns
    -------
    list
        The count rates for DR, DL, AR, and AL coincidences, in order.
    list
        The uncertainties for DR, DL, AR, and AL coincidences, in order.
    '''
    m.meas_basis("DR")
    DR, DR_unc = m.take_data(*samp, "C4")
    m.meas_basis("DL")
    DL, DL_unc = m.take_data(*samp, "C4")
    m.meas_basis("AR")
    AR, AR_unc = m.take_data(*samp, "C4")
    m.meas_basis("AL")
    AL, AL_unc = m.take_data(*samp, "C4")
    return [DR, DL, AR, AL], [DR_unc, DL_unc, AR_unc, AL_unc]

def meas_all(m:Manager, samp:Tuple[int, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    ''' Measure alpha, beta, and phi parameters of the state.

    Parameters
    ----------
    m : Manager
        The manager object running the experiment.
    samp : Tuple[int, float]
        The sampling parameters for measurements.
    
    Returns
    -------
    Tuple[float, float, float]
        The alpha, beta, and phi parameters of the state.
    Tuple[float, float, float]
        Uncertainties in the alpha, beta, and phi parameters.
    '''
    # take data in H/V basis
    (HH, HV, VH, VV), (HHu, HVu, VHu, VVu) = meas_HV(m, samp)
    T_hv = HH + HV + VH + VV
    
    # take data in DA/RL basis
    (DR, DL, AR, AL), (DRu, DLu, ARu, ALu) = meas_DA_RL(m, samp)
    T_dr = DR + DL + AR + AL

    # compute alpha
    alpha = np.arctan(np.sqrt((VH+VV)/(HH+HV)))
    # compute alpha uncertainty
    alpha_unc = 1/(2*T_hv) * np.sqrt((HHu**2 + HVu**2) * (VH+VV)/(HH+HV) + (VHu**2 + VVu**2) * (HH+HV)/(VH+VV))

    # compute beta
    beta = np.arctan(np.sqrt((HV+VH)/(HH+VV)))
    # compute uncertainty in beta
    beta_unc = 1/(2*T_hv) * np.sqrt((HHu**2 + VVu**2) * (VH+HV)/(HH+VV) + (HVu**2 + VHu**2) * (HH+VV)/(HV+VH))
    
    # compute phi
    u = (DR-DL-AR+AL)/(T_dr*np.sin(2*alpha))
    phi = np.arcsin(u)

    # compute uncertainty in phi
    phi_unc = 2/(np.sqrt(1-u**2)*np.sin(2*alpha)) * np.sqrt((DL + AR)/T_dr * (DRu**2 + ALu**2) + (DR + AL)/T_dr * (DLu**2 + ARu**2) + u**2*np.cos(2*alpha)**2*alpha_unc**2)

    # return it all!
    return (alpha, beta, phi), (alpha_unc, beta_unc, phi_unc)


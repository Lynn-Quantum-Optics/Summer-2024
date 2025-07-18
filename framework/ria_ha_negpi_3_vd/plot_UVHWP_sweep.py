'''
Author: Ria Haapala
Last Update: 7/2/2025

This file can be used to load and plot the UVHWP sweep data generated in a data collection of six chi values.
The file will plot the data and generate a list of chi values, which are identical to the values the data 
collection file will generate itself.
'''

from lab_framework import Manager, analysis
import numpy as np
import scipy.optimize as opt # type: ignore
from analysis_old import *
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat


if __name__ == '__main__':
    TRIAL = 2
    chi_vals = np.linspace(0.001, np.pi/2, 6)
    chi = chi_vals[1]

    # obtain the first round of data and switch to a new output file
    data1 = pd.read_csv(f"UVHWP_calibration/ha_sweep_{chi}_{TRIAL}.csv")
    data2 = pd.read_csv(f'UVHWP_calibration/vd_sweep_{chi}_{TRIAL}.csv')

    args1, unc1 = fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_SEM)
    args2, unc2 = fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_SEM)

    # Calculate the UVHWP angle we want for a given CHI value
    desired_ratio = (np.cos(chi/2) / np.sin(chi/2))**2
    def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple):
        ''' Function want to minimize'''
        return (sin2_sq(x_, *args1_) / sin2_sq(x_, *args2_) - desired_ratio)**2
    x_min, x_max = np.min(data1.C_UV_HWP), np.max(data1.C_UV_HWP)
    UVHWP_angle = opt.brute(min_me, args=(args1, args2), ranges=((x_min, x_max),))

    # might need to retune this if there are multiple roots. I'm only assuming one root
    print(UVHWP_angle)
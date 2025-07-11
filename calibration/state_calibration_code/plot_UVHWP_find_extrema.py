from lab_framework import Manager, analysis
import numpy as np
import scipy.optimize as opt # type: ignore
from analysis_old import *
import pandas as pd
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat

# similar to the plot_UVHWP file (which load tomography UVHWP sweep data and plots it), this file uses the basic_ratio_tuning output sweeps to
# to calculate where chi=0 and chi=pi/2 occur. It also replots the data

if __name__ == '__main__':
    mpName = "ria"
    date = "07092025"
    TRIAL = 1

    # manually perform sweep of UVHWP for various chi values (chi/2=eta)
    chi_vals = [0.001, np.pi-0.001]

    # obtain the first round of data and switch to a new output file
    data1 = pd.read_csv(f"vv_sweep_{TRIAL}.csv")
    data2 = pd.read_csv(f'hh_sweep_{TRIAL}.csv')

    args1, unc1 = fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_SEM)
    args2, unc2 = fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_SEM)

    for chi in chi_vals:
        # Calculate the UVHWP angle we want for a given CHI value
        desired_ratio = (np.sin(chi/2) / np.cos(chi/2))**2
        print(desired_ratio)
        def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple):
            ''' Function want to minimize'''
            return (sin2_sq(x_, *args1_) / sin2_sq(x_, *args2_) - desired_ratio)**2
        x_min, x_max = np.min(data1.C_UV_HWP), np.max(data1.C_UV_HWP)
        UVHWP_angle = opt.brute(min_me, args=(args1, args2), ranges=((x_min, x_max),))

        # might need to retune this if there are multiple roots. I'm only assuming one root
        print(UVHWP_angle)

    df1 = Manager.load_data(f"vv_sweep_{TRIAL}.csv")
    df2 = Manager.load_data(f"hh_sweep_{TRIAL}.csv")


    angles1, gammas1 = df1['C_UV_HWP'], df1['C4']
    angles2, gammas2 = df2['C_UV_HWP'], df2['C4']

    thetas = unp.arctan(unp.sqrt((gammas1)/(gammas2)))

    params = analysis.fit('line', angles1, thetas)
    analysis.plot_func('line', params, angles1, color='blue')
    analysis.plot_errorbar(angles1, thetas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Theta Parameter (rad)')
    plt.savefig(f'UVHWP_plot_{TRIAL}.png', dpi=600)
    plt.show()

    x = analysis.find_value('line', params, np.pi/4, angles1)
    print(f'Pi/4 at {x}')

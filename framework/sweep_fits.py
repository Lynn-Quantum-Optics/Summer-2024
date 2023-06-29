from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import Manager
from measurement import meas_ab, meas_phi, meas_all
from core import analysis

def sweep_qp_phi(m:Manager, samp:Tuple[int,float], num_step:int, out_file:str=None):
    '''
    Sweep the quartz-plate to record the phi parameter as a function of angle.
    '''

    # array of angles (mostly for plotting)
    pos_min = 0
    pos_max = 45
    angles = np.linspace(pos_min, pos_max, num_step)

    # sweeps in different bases
    m.meas_basis('DR')
    DR, DRu = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('DL')
    DL, DLu = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('RR')
    RR, RRu = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('RL')
    RL, RLu = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)

    # compute phi
    u = (DL-DR)/(RR-RL)
    phis = np.arctan2((DL-DR), (RR-RL))
    phi_uncs = 1/(1+u**2) * 1/np.abs(RR-RL) * np.sqrt(DLu**2 + DRu**2 + u**2 * (RRu**2 + RLu**2))
    
    # repackage data into a dataframe
    df = pd.DataFrame({'QP':angles, 'phi':phis, 'unc':phi_uncs})

    # make phi > 0 at 0 degrees
    df['phi'] += 360

    # this is a silly bit of code removes any discontinuities
    for i in range(1, len(df)):
        if abs(df['phi'][i] - df['phi'][i-1]) > 180:
            df['phi'][i:] -= 360

    # save the data if an output file was provided
    if out_file is not None:
        df.to_csv(out_file, index=False)

    # error bar plot of the data
    plt.errorbar(df['QP'], df['phi'], df['unc'], fmt='ro', ms=0.1)

    # fit a secant function to the datas!
    params, uncs = analysis.fit('sec', df['QP'], df['phi'], df['unc'], p0=[-1000, 500], bounds=([-1e5, -2000], [0, 2000]))
    analysis.plot_func('sec', params, df['QP'], label=f'${params[0]:.3f}\\sec(\\theta) + {params[1]:.3f}$')
    
    # plot axes and such
    plt.xlabel('Quartz Plate Angle (degrees)')
    plt.ylabel('Phi Parameter (degrees)')
    plt.title(f'Phi parameter as quartz plate\nis swept from {df["QP"].min()} to {df["QP"].max()} degrees')
    plt.legend()

    # print the results and show the plot
    print(f'a = {params[0]} ± {uncs[0]}')
    print(f'b = {params[1]} ± {uncs[1]}')
    plt.show()

def sweep_uvhwp_alph(m:Manager, samp:Tuple[int,float], num_step:int, out_file:str=None):
    '''
    Sweep the UVHWP while collecting alpha data.
    '''
    # overall collection parameters
    pos_min = -20
    pos_max = 50

    # array of angles (mostly for plotting later)
    angles = np.linspace(pos_min, pos_max, num_step)

    # gather coincidence data in the four relevant bases
    m.meas_basis('HH')
    HH, HHu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('HV')
    HV, HVu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VH')
    VH, VHu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VV')
    VV, VVu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)

    # total count rates
    T = HH + HV + VH + VV

    # compute alpha
    alpha = np.arctan(np.sqrt((VH+VV)/(HH+HV)))
    alpha_unc = 1/(2*T) * np.sqrt((HHu**2 + HVu**2) * (VH+VV)/(HH+HV) + (VHu**2 + VVu**2) * (HH+HV)/(VH+VV))
    
    # using degrees from here on out
    alpha = np.rad2deg(alpha)
    alpha_unc = np.rad2deg(alpha_unc)

    # repackage data into a dataframe
    df = pd.DataFrame({'UVHWP':angles, 'alpha':alpha, 'unc':alpha_unc})

    # save if requested
    if out_file is not None:
        df.to_csv(out_file, index=False)

    # error bar plot of the data
    plt.errorbar(df['UVHWP'], df['alpha'], df['unc'], fmt='ro', ms=0.1)

    # fit a secant function to the datas!
    # params, uncs = analysis.fit('sin2', df['QP'], df['phi'], df['unc'], p0=[-1000, 500], bounds=([-1e5, -2000], [0, 2000]))
    # analysis.plot_func('sec', params, df['QP'], label=f'${params[0]:.3f}\\sec(\\theta) + {params[1]:.3f}$')
    
    # plot axes and such
    plt.xlabel('UV Half Wave Plate Angle (degrees)')
    plt.ylabel('Alpha Parameter (degrees)')
    plt.title(f'Alpha parameter as UVHWP plate\nis swept from {df["UVHWP"].min()} to {df["UVHWP"].max()} degrees')
    plt.legend()

    # print the results and show the plot
    # print(f'a = {params[0]} ± {uncs[0]}')
    # print(f'b = {params[1]} ± {uncs[1]}')
    plt.show()

if __name__ == '__main__':
    
    SAMP = (3,0.5)
    NUM_STEP =50

    m = Manager()
    m.make_state("phi_plus")
    # sweep_qp_phi(None, SAMP, NUM_STEP, None)
    sweep_uvhwp_alph(m, SAMP, NUM_STEP, 'uvhwp_alph_sweep.csv')

    m.shutdown()
    
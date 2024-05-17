''' stray_count_checking.py

This file runs an experiment and analysis that will help us determine if the stray HV and VH counts that we are observing are the product of Bob's creation HWP or the measurement waveplates being misaligned, or if they are the result from naturally noisy data (i.e. other light sources hitting the detectors).

The idea here is that if HV and VH counts come randomly, then HV/VH will be random as well. However, if we are creating the HV and VH counts, then HV/VH will be correlated with HH/VV.
'''
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core import Manager

def run_sweeps(out_file:str, start_angle:float, end_angle:float, num_steps:int, samp:Tuple[int, float], state_preset:str=None) -> pd.DataFrame:
    ''' Runs a sweep of the UVHWP while taking data in the HH, HV, VH, and VV bases.

    Parameters
    ----------
    out_file : str
        The file to save the data to.
    start_angle : float
        The angle to start the UVHWP at.
    end_angle : float
        The angle to end the UVHWP at.
    num_steps : int
        The number of steps to take between start_angle and end_angle.
    samp : Tuple[int, float]
        Sampling parameters for measurements.
    state_preset : str, optional
        The state preset to use. If None, then no state preset is used.
    
    Returns
    -------
    pd.DataFrame
        The dataframe that was saved.
    '''

    # initialize the manager
    m = Manager()

    # set the state preset if one was given
    if state_preset is not None:
        m.make_state(state_preset)
    
    # initialize the dataframe
    df = pd.DataFrame({'UVHWP': np.linspace(start_angle, end_angle, num_steps)})

    m.meas_basis('HH')
    df['HH'], df['HHu'] = m.sweep('C_UV_HWP', start_angle, end_angle, num_steps, *samp)

    m.meas_basis('HV')
    df['HV'], df['HVu'] = m.sweep('C_UV_HWP', start_angle, end_angle, num_steps, *samp)

    m.meas_basis('VH')
    df['VH'], df['VHu'] = m.sweep('C_UV_HWP', start_angle, end_angle, num_steps, *samp)

    m.meas_basis('VV')
    df['VV'], df['VVu'] = m.sweep('C_UV_HWP', start_angle, end_angle, num_steps, *samp)

    # save the data frame
    df.to_csv(out_file, index=False)

    # shutdown the manager
    m.shutdown()

    # return the dataframe
    return df

def make_plots(data:pd.DataFrame, plot_outfile:str=None) -> None:
    ''' Generates plots of the data collected from run_sweeps.
    '''
    # calculate count rate ratios and uncertainties
    HH_VV = data['HH']/data['VV']
    HH_VV_unc = np.sqrt((data['HHu']/data['VV'])**2 + (data['HH']*data['VVu']/data['VV']**2)**2)

    HV_VH = data['HV']/data['VH']
    HV_VH_unc = np.sqrt((data['HVu']/data['VH'])**2 + (data['HV']*data['VHu']/data['VH']**2)**2)

    # initialize the figure
    fig, ax = plt.subplots(2,1,sharex=True)

    # plot the count rate ratios
    ax[0].errorbar(data['UVHWP'], HH_VV, yerr=HH_VV_unc, fmt='o', ms=0.1)
    ax[1].errorbar(data['UVHWP'], HV_VH, yerr=HV_VH_unc, fmt='o', ms=0.1)

    # titles and such
    ax[0].set_ylabel('HH/VV')
    ax[1].set_ylabel('HV/VH')
    ax[1].set_xlabel('UVHWP Angle (degrees)')
    fig.suptitle('HH/VV and HV/VH Count Rate Ratios\nas a Function of UVHWP Angle')

    # save plot if requested
    if plot_outfile:
        plt.savefig(plot_outfile, dpi=600)

    # show plot
    plt.show()

if __name__ == '__main__':
    data = run_sweeps(
        out_file='data/stray_count_checking/sweep_data.csv',
        start_angle=-10,
        state_preset='phi_plus',
        end_angle=50,
        num_steps=30,
        samp=(5,1))
    # make_plots(data, 'data/stray_count_checking/count_ratio_plots.png')
    # make_plots(pd.read_csv('data/stray_count_checking/sweep_data.csv'), 'data/stray_count_checking/count_ratio_plots.png')



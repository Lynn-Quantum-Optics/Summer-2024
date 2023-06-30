from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import Manager
from core import analysis

# +++ DATA COLLECTION FUNCTIONS +++

def sweep_qp_phi(m:Manager, pos_min:float, pos_max:float, num_step:int, samp:Tuple[int,float], out_file:str) -> None:
    ''' Sweep the quartz-plate to record the phi parameter as a function of angle.

    The output of this function is a csv file with the columns:
    - 'QP' the angle of the quartz-plate
    - 'phi' the phi parameter
    - 'unc' the uncertainty in the phi parameter

    Parameters
    ----------
    m : Manager
        The manager object that is controlling the experiment.
    pos_min : float
        The minimum position of the quartz-plate.
    pos_max : float
        The maximum position of the quartz-plate.
    num_step : int
        The number of steps to take between pos_min and pos_max.
    samp : Tuple[int,float]
        The sample parameters for each measurement.
    out_file : str
        The name of the file to save the data to.
    '''

    # array of angles (mostly for plotting)
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
    phis = np.rad2deg(np.arctan2((DL-DR), (RR-RL)))
    phi_uncs = np.rad2deg(1/(1+u**2) * 1/np.abs(RR-RL) * np.sqrt(DLu**2 + DRu**2 + u**2 * (RRu**2 + RLu**2)))
    
    # repackage data into a dataframe
    df = pd.DataFrame({'QP':angles, 'phi':phis, 'unc':phi_uncs})

    # this is a silly bit of code removes any discontinuities
    for i in range(1, len(df)):
        if abs(df['phi'][i-1] - df['phi'][i]) > 180:
            df['phi'][i:] += 360

    # save the data
    df.to_csv(out_file, index=False)

def sweep_uvhwp_alph(m:Manager, pos_min:float, pos_max:float, num_step:int, samp:Tuple[int,float], out_file:str) -> None:
    ''' Sweep the UVHWP across a range of angles while measuring the alpha parameter.
    
    The output of this function is a csv file with the columns:
    - 'UVHWP' the angle of the UVHWP
    - 'alpha' the alpha parameter
    - 'unc' the uncertainty in the alpha parameter

    Parameters
    ----------
    m : Manager
        The manager object that is controlling the experiment.
    pos_min : float
        The minimum position of the UVHWP.
    pos_max : float
        The maximum position of the UVHWP.
    num_step : int
        The number of steps to take between pos_min and pos_max.
    samp : Tuple[int,float]
        The sample parameters for each measurement.
    out_file : str
        The name of the file to save the data to.
    '''
    
    # overall collection parameters
    pos_min = -10
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

def sweep_bchwp_beta(m:Manager, pos_min:float, pos_max:float, num_step:int, samp:Tuple[int,float], out_file:str) -> None:
    '''Sweep Bob's HWP across a range of angles while measuring the beta parameter.
    
    The output of this function is a csv file with the columns:
    - 'BCHWP' the angle of BCHWP
    - 'beta' the beta parameter
    - 'unc' the uncertainty in the beta parameter

    Parameters
    ----------
    m : Manager
        The manager object that is controlling the experiment.
    pos_min : float
        The minimum position of the UVHWP.
    pos_max : float
        The maximum position of the UVHWP.
    num_step : int
        The number of steps to take between pos_min and pos_max.
    samp : Tuple[int,float]
        The sample parameters for each measurement.
    out_file : str
        The name of the file to save the data to.
    '''
    # overall collection parameters
    pos_min = -10
    pos_max = 50

    # array of angles (mostly for plotting later)
    angles = np.linspace(pos_min, pos_max, num_step)

    # gather coincidence data in the four relevant bases
    m.meas_basis('HH')
    HH, HHu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('HV')
    HV, HVu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VH')
    VH, VHu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VV')
    VV, VVu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)
    
    # total count rates
    T = HH + HV + VH + VV

    # compute beta
    beta = np.rad2deg(np.arctan(np.sqrt((HV+VH)/(HH+VV))))
    beta_unc = np.rad2deg(1/(2*T) * np.sqrt((HHu**2 + VVu**2) * (VH+HV)/(HH+VV) + (HVu**2 + VHu**2) * (HH+VV)/(HV+VH)))

    # repackage data into a dataframe
    df = pd.DataFrame({'BCHWP':angles, 'beta':beta, 'unc':beta_unc})

    # save
    df.to_csv(out_file, index=False)

# +++ DATA DISPLAY +++

def show_data(fp:str):
    ''' Show the data that was collected using one of the sweep methods.
    
    Displays a basic errorbar plot with no fitting.

    Parameters
    ----------
    fp : str
        Path to the data file.
    '''

    # open data frame
    df = pd.read_csv(fp)

    # get component and parameter
    comp, param, _ = df.columns

    plt.errorbar(df[comp], df[param], df['unc'], fmt='ro', ms=0.1)
    plt.title(f'{param} as {comp} is\nswept from {df[comp].min():.1f} to {df[comp].max():.1f}')
    plt.xlabel(f'{comp} position (degrees)')
    plt.ylabel(f'{param} parameter (degrees)')
    plt.show()

# +++ DATA FITTING +++

def fit_phi(fp:str) -> None:
    ''' Perform a secant fit for the quartz plate-phi sweep.

    Parameters
    ----------
    fp : str
        Path to the data file.
    '''
    # load the data from a file
    df = pd.read_csv(fp)

    # error bar plot of the data
    plt.errorbar(df['QP'], df['phi'], df['unc'], fmt='ro', ms=0.1)

    # fit a secant function to the data
    # but only include up to phi(0)+360
    df = df[df['phi'] <= df['phi'][0]+360]
    params, uncs = analysis.fit('sec', df['QP'], df['phi'], df['unc'], p0=[1000, 1, -1000], maxfev=10000)
    analysis.plot_func('sec', params, df['QP'], label=f'${params[0]:.3f}\\sec({params[1]:.3f} x) + {params[2]:.3f}$')
    
    # plot axes and such
    plt.xlabel('Quartz Plate Angle (degrees)')
    plt.ylabel('Phi Parameter (degrees)')
    plt.title(f'Phi parameter as quartz plate\nis swept from {df["QP"].min()} to {df["QP"].max()} degrees')
    plt.legend()

    # print the results and show the plot
    print(f'"a": {params[0]} ± {uncs[0]}')
    print(f'"b": {params[1]} ± {uncs[1]}')
    print(f'"c": {params[2]} ± {uncs[2]}')
    plt.show()

def fit_ab(fp:str, linear_region:Tuple[float, float]):
    ''' Perform linear fits for alpha/beta data.
    
    Parameters
    ----------
    fp : str
        Path to the data file.
    linear_region : Tuple[float, float]
        Ordered pair of angles that define the linear region of the plot.
    '''
    # open data frame
    df = pd.read_csv(fp)

    # get component and parameter
    comp, param, _ = df.columns

    # fit a line in the linear region
    dfl = df[(linear_region[0] < df[comp])*(df[comp] < linear_region[1])]
    fit_params, fit_uncs = analysis.fit('line', dfl[comp], dfl[param], dfl['unc'])
    
    # find extremal values
    max_i = df[param].idxmax()
    max_x = df[comp][max_i]
    max_y = df[param][max_i]
    min_i = df[param].idxmin()
    min_x = df[comp][min_i]
    min_y = df[param][min_i]

    # plot stuff
    analysis.plot_func('line', fit_params, df[comp], label=f'$y={fit_params[0]:.2f}x + {fit_params[1]:.2f}$', color='b')
    plt.errorbar(df[comp], df[param], df['unc'], ms=0.1, label='Data', fmt='ro')
    
    plt.title(f'{param} as {comp} is\nswept from {df[comp].min():.1f} to {df[comp].max():.1f}')
    plt.xlabel(f'{comp} position (degrees)')
    plt.ylabel(f'{param} parameter (degrees)')
    plt.legend()

    print(f'"m": {fit_params[0]}')
    print(f'"b": {fit_params[1]}')
    print(f'"{comp}_min": {min_x}')
    print(f'"{param}_min": {min_y}')
    print(f'"{comp}_max": {max_x}')
    print(f'"{param}_max": {max_y}')

    plt.show()

# +++ MAIN SCRIPT +++
if __name__ == '__main__':
    
    SAMP = (5, 0.5)
    NUM_STEP = 50

    # m = Manager()
    # m.make_state("phi_plus")
    # sweep_qp_phi(None, SAMP, NUM_STEP, 'qp_phi_sweep_no_discontinuities.csv')
    # sweep_uvhwp_alph(m, SAMP, NUM_STEP, 'uvhwp_alph_sweep.csv')
    # sweep_uvhwp_alph(None, SAMP, NUM_STEP, 'uvhwp_alph_sweep.csv')
    # sweep_bchwp_beta(m, SAMP, NUM_STEP, 'bchwp_beta_sweep.csv')
    # sweep_bchwp_beta(None, SAMP, NUM_STEP, 'bchwp_beta_sweep.csv')
    # m.shutdown()

    # show_data('bchwp_beta_sweep.csv')
    fit_ab('bchwp_beta_sweep.csv', (10, 35))
    # fit_ab('uvhwp_alph_sweep.csv', (5, 35))


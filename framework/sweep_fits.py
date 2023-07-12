from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import Manager
from core import analysis

# +++ PARAMETER CALCULATION FUNCTIONS +++

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
    beta = np.arcsin(u)/2
    # compute beta uncertainty
    unc = 1/(2*np.sqrt(1-u**2)) * np.sqrt(HDu**2 + HAu**2 + VDu**2 + VAu**2)

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

# +++ DATA COLLECTION FUNCTIONS +++

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

    # array of angles (mostly for plotting later)
    angles = np.linspace(pos_min, pos_max, num_step)

    # gather coincidence data in the four relevant bases
    m.meas_basis('HD')
    HD, HDu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('HA')
    HA, HAu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VD')
    VD, VDu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VA')
    VA, VAu = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)

    # compute alpha
    alpha, alpha_unc = calc_alpha((HD, HA, VD, VA), (HDu, HAu, VDu, VAu))

    # repackage data into a dataframe
    df = pd.DataFrame({'UVHWP':angles, 'alpha':alpha, 'unc':alpha_unc, 'HD':HD, 'HA':HA, 'VD':VD, 'VA':VA, 'HDu':HDu, 'HAu':HAu, 'VDu':VDu, 'VAu':VAu})

    # save
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

    # array of angles (mostly for plotting later)
    angles = np.linspace(pos_min, pos_max, num_step)

    # gather coincidence data in the four relevant bases
    m.meas_basis('HD')
    HD, HDu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('HA')
    HA, HAu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VD')
    VD, VDu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('VA')
    VA, VAu = m.sweep('B_C_HWP', pos_min, pos_max, num_step, *samp)

    # compute alpha
    beta, beta_unc = calc_beta((HD, HA, VD, VA), (HDu, HAu, VDu, VAu))

    # repackage data into a dataframe
    df = pd.DataFrame({'BCHWP':angles, 'beta':beta, 'unc':beta_unc, 'HD':HD, 'HA':HA, 'VD':VD, 'VA':VA, 'HDu':HDu, 'HAu':HAu, 'VDu':VDu, 'VAu':VAu})

    # save
    df.to_csv(out_file, index=False)

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
    phis, phi_uncs = calc_phi((DL, DR, RR, RL), (DRu, DRu, RRu, RLu))

    # repackage data into a dataframe
    df = pd.DataFrame({'QP':angles, 'phi':phis, 'unc':phi_uncs, 'DL':DL, 'DR':DR, 'RR':RR, 'RL':RL, 'DLu':DLu, 'DRu':DRu, 'RRu':RRu, 'RLu':RLu})

    # this is a silly bit of code removes any discontinuities
    for i in range(1, len(df)):
        if abs(df['phi'][i-1] - df['phi'][i]) > 180:
            df['phi'][i:] += 360

    # save the data
    df.to_csv(out_file, index=False)

# +++ DATA DISPLAY +++

def show_data(fp:str, plot_outfile:str=None):
    ''' Show the data that was collected using one of the sweep methods.
    
    Displays a basic errorbar plot with no fitting.

    Parameters
    ----------
    fp : str
        Path to the data file.
    plot_outfile : str (optional, default None)
        Output file name for the plot created.
    '''

    # open data frame
    df = pd.read_csv(fp)

    # get component and parameter
    comp, param = df.columns[:2]

    plt.errorbar(df[comp], df[param], df['unc'], fmt='ro', ms=0.1)
    plt.title(f'{param} as {comp} is\nswept from {df[comp].min():.1f} to {df[comp].max():.1f}')
    plt.xlabel(f'{comp} position (degrees)')
    plt.ylabel(f'{param} parameter (degrees)')

    if plot_outfile:
        plt.savefig(plot_outfile, dpi=600)
    plt.show()

# +++ DATA FITTING +++

def fit_phi(fp:str, plot_outfile:str=None) -> None:
    ''' Perform a secant fit for the quartz plate-phi sweep.

    Parameters
    ----------
    fp : str
        Path to the data file.
    plot_outfile : str (optional, default None)
        Output file name for the plot created.
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

    if plot_outfile:
        plt.savefig(plot_outfile, dpi=600)
    plt.show()

def fit_ab(fp:str, linear_region:Tuple[float, float], plot_outfile:str=None):
    ''' Perform linear fits for alpha/beta data.
    
    Parameters
    ----------
    fp : str
        Path to the data file.
    linear_region : Tuple[float, float]
        Ordered pair of angles that define the linear region of the plot.
    plot_outfile : str (optional, default None)
        Output file name for the plot created.
    '''
    # open data frame
    df = pd.read_csv(fp)

    # get component and parameter
    comp, param = df.columns[:2]

    # fit a line in the linear region
    dfl = df[(linear_region[0] < df[comp]) & (df[comp] < linear_region[1])]
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

    print(f'"m": {fit_params[0]} ± {fit_uncs[0]}')
    print(f'"b": {fit_params[1]} ± {fit_uncs[1]}')
    print(f'"{comp}_min": {min_x}')
    print(f'"{param}_min": {min_y}')
    print(f'"{comp}_max": {max_x}')
    print(f'"{param}_max": {max_y}')
    
    if plot_outfile:
        plt.savefig(plot_outfile, dpi=600)
    plt.show()

# +++ MAIN SCRIPT +++
if __name__ == '__main__':
    # +++ hyper parameters
    SAMP = (5, 1)
    NUM_STEP = 50

    # +++ UVHWP alpha sweep experiment
    ''' 
    m = Manager()
    m.make_state("phi_plus")
    sweep_uvhwp_alph(m, -10, 50, NUM_STEP, SAMP, 'uvhwp_alpha_sweep.csv')
    m.shutdown()
    show_data('uvhwp_alpha_sweep.csv', 'uvhwp_alpha_sweep_nofit2.png')
    '''

    # +++ UVHWP data fitting
    '''
    fit_ab('uvhwp_alpha_sweep.csv', (3,40), 'uvhwp_alpha_sweep.png')
    '''
    
    # +++ BCHWP beta sweep experiment
    '''
    m = Manager()
    m.make_state("phi_plus")
    sweep_bchwp_beta(m, -10, 50, NUM_STEP, SAMP, 'bchwp_beta_sweep.csv')
    m.shutdown()
    show_data('bchwp_beta_sweep.csv', 'bchwp_beta_sweep_nofit.png')
    '''

    # +++ QP phi sweep experiment
    m = Manager()
    m.make_state("phi_plus")
    sweep_qp_phi(m, -45, 0, NUM_STEP, SAMP, 'qp_phi_sweep_neg.csv')
    m.shutdown()
    # show_data('qp_phi_sweep.csv', 'qp_phi_sweep_nofit.png')
    
    
    # sweep_qp_phi(None, SAMP, NUM_STEP, 'qp_phi_sweep_no_discontinuities.csv')
    # sweep_uvhwp_alph(None, SAMP, NUM_STEP, 'uvhwp_alph_sweep.csv')
    # sweep_bchwp_beta(None, SAMP, NUM_STEP, 'bchwp_beta_sweep.csv')
    

    # show_data('uvhwp_alpha_sweep.csv')


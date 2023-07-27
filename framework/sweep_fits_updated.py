from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from lab_framework import Manager, analysis
from uncertainties import ufloat
from uncertainties import unumpy as unp
from full_tomo_updated_richard import *

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

def calc_phi(DL_DR_RR_RL):
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

    # get total count rates and convert everything to expectation values
    T = DL + DR + RR + RL
    DL, DR, RR, RL = DL/T, DR/T, RR/T, RL/T
        # calculate phi
    phi = unp.arctan2(DL-DR, RR-RL)

    return np.rad2deg([x.nominal_value for x in phi]), np.rad2deg([x.std_dev for x in phi])

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

    m.configure_motors(
        C_UV_HWP = 22.5,
        B_C_HWP=0
    )
    # array of angles (mostly for plotting)
    angles = np.linspace(pos_min, pos_max, num_step)

    # sweeps in different bases
    # sweep returns angles swept over and then ufloats for counts and unc
    m.meas_basis('DR')
    _, DR = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('DL')
    _, DL = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('RR')
    _, RR = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('RL')
    _, RL = m.sweep('C_QP', pos_min, pos_max, num_step, *samp)

    print(DR, DL, RR, RL)

    # compute phi
    phis, phi_uncs = calc_phi((DL, DR, RR, RL))

    # repackage data into a dataframe
    df = pd.DataFrame({'QP':angles, 'phi':phis, 'unc':phi_uncs, 'DL':[dl.nominal_value for dl in DL], 'DR':[dr.nominal_value for dr in DR], 'RR':[rr.nominal_value for rr in RR], 'RL':[rl.nominal_value for rl in RL], 'DLu':[dl.std_dev for dl in DL], 'DRu':[dr.std_dev for dr in DR], 'RRu':[rr.std_dev for rr in RR], 'RLu':[rl.std_dev for rl in RL]})

    # this is a silly bit of code removes any discontinuities
    for i in range(1, len(df)):
        if abs(df['phi'][i-1] - df['phi'][i]) > 180:
            df['phi'][i:] += 360

    # save the data
    df.to_csv(out_file, index=False)

    # make quick plot
    plt.errorbar(df['QP'].values, df['phi'].values, df['unc'].values, fmt='o')
    plt.xlabel('QP')
    plt.ylabel('$\phi$')
    plt.title('$\phi$ dependence on QP')
    plt.savefig(out_file.split('.')[0]+'.pdf')
    plt.show()

def sweep_uvhwp_phi(m:Manager, pos_min:float, pos_max:float, num_step:int, samp:Tuple[int,float], out_file:str) -> None:
    ''' Sweep the UV_HWP to record the phi parameter as a function of angle.

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
    m.configure_motors(
        C_QP = 0,
        B_C_HWP=0
    )
    # array of angles (mostly for plotting)
    angles = np.linspace(pos_min, pos_max, num_step)

    # sweeps in different bases
    # sweep returns angles swept over and then ufloats for counts and unc
    m.meas_basis('DR')
    _, DR = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('DL')
    _, DL = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('RR')
    _, RR = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)
    m.meas_basis('RL')
    _, RL = m.sweep('C_UV_HWP', pos_min, pos_max, num_step, *samp)

    print(DR, DL, RR, RL)

    # compute phi
    phis, phi_uncs = calc_phi((DL, DR, RR, RL))

    # repackage data into a dataframe
    df = pd.DataFrame({'UV_HWP':angles, 'phi':phis, 'unc':phi_uncs, 'DL':[dl.nominal_value for dl in DL], 'DR':[dr.nominal_value for dr in DR], 'RR':[rr.nominal_value for rr in RR], 'RL':[rl.nominal_value for rl in RL], 'DLu':[dl.std_dev for dl in DL], 'DRu':[dr.std_dev for dr in DR], 'RRu':[rr.std_dev for rr in RR], 'RLu':[rl.std_dev for rl in RL]})

    # this is a silly bit of code removes any discontinuities
    for i in range(1, len(df)):
        if abs(df['phi'][i-1] - df['phi'][i]) > 180:
            df['phi'][i:] += 360

    # save the data
    df.to_csv(out_file, index=False)

    # make quick plot
    plt.errorbar(df['UV_HWP'].values, df['phi'].values, df['unc'].values, fmt='o')
    plt.xlabel('UVHWP')
    plt.ylabel('$\phi$')
    plt.title('$\phi$ dependence on UVHWP')
    plt.savefig(out_file.split('.')[0]+'.pdf')
    plt.show()

def sweep_uvhwp_qp_phi(m, UV_min, UV_max, QP_min, QP_max, num_steps, Samp, out_file):
    '''Sweep QP and UVHWP simultaneously and measure phi'''
    UV_pos = np.linspace(UV_min, UV_max, num_steps)
    QP_pos = np.linspace(QP_min, QP_max, num_steps)
    def multi_sweep():
        # open output
        out = []
        # loop to perform the sweep
        for i in trange(num_steps):
            uv = UV_pos[i]
            qp = QP_pos[i]
            m.configure_motors(
                C_UV_HWP = uv,
                C_QP = qp
            )
            x = m.take_data(Samp[0], Samp[1], Manager.MAIN_CHANNEL)
            out.append(x)
        return np.array(out)
    # sweeps in different bases
    # sweep returns angles swept over and then ufloats for counts and unc
    m.meas_basis('DR')
    _, DR = multi_sweep()
    m.meas_basis('DL')
    _, DL = multi_sweep()
    m.meas_basis('RR')
    _, RR = multi_sweep()
    m.meas_basis('RL')
    _, RL = multi_sweep()

    print(DR, DL, RR, RL)

    # compute phi
    phis, phi_uncs = calc_phi((DL, DR, RR, RL))

    # repackage data into a dataframe
    df = pd.DataFrame({'UV_HWP':UV_pos, 'QP':QP_pos, 'phi':phis, 'unc':phi_uncs, 'DL':[dl.nominal_value for dl in DL], 'DR':[dr.nominal_value for dr in DR], 'RR':[rr.nominal_value for rr in RR], 'RL':[rl.nominal_value for rl in RL], 'DLu':[dl.std_dev for dl in DL], 'DRu':[dr.std_dev for dr in DR], 'RRu':[rr.std_dev for rr in RR], 'RLu':[rl.std_dev for rl in RL]})

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

def sweep_full_tomo(UV_HWP_ls=None, QP_ls=None):
    '''Get full tomo for all states in sweeps of diff angles. For each, list of min, max, step'''
    angles_init = [22.5, 0, 0]
    m.configure_motors(
        C_UV_HWP=angles_init[0],
        C_QP = angles_init[1],
        B_C_HWP=angles_init[2]
    )
    
    if QP_ls is None and UV_HWP_ls is not None:
        UV_angles = np.linspace(UV_HWP_ls[0], UV_HWP_ls[1], UV_HWP_ls[2])
        for uv in UV_angles:
            m.C_UV_HWP.goto(uv)
            rho, unc, Su, un_proj, un_proj_unc = get_rho(m, SAMP)
            angles_init[0] = uv

            # save results
            with open(f'decomp_test/rho_uv_{uv}.npy', 'wb') as f:
                np.save(f, (rho, unc, Su, un_proj, un_proj_unc, angles_init))
    elif QP_ls is not None and UV_HWP_ls is None:
        QP_angles = np.linspace(QP_ls[0], QP_ls[1], QP_ls[2])
        for qp in QP_angles:
            m.C_QP.goto(qp)
            rho, unc, Su, un_proj, un_proj_unc = get_rho(m, SAMP)
            angles_init[1] = qp

            # save results
            with open(f'decomp_test/rho_qp_{qp}.npy', 'wb') as f:
                np.save(f, (rho, unc, Su, un_proj, un_proj_unc, angles_init))
    else:
        UV_angles = np.linspace(UV_HWP_ls[0], UV_HWP_ls[1], UV_HWP_ls[2])
        QP_angles = np.linspace(QP_ls[0], QP_ls[1], QP_ls[2])
        assert len(UV_angles) == len(QP_angles), f'lists must have same length! uvhwp: {len(UV_angles)}, qp: {len(QP_angles)}'
        for i in range(len(UV_angles)):
            uv = UV_angles[i]
            qp = QP_angles[i]

            m.configure_motors(
                C_UV_HWP = uv,
                C_QP = qp
            )

            angles_init[0] = uv
            angles_init[1] = qp

            # save results
            with open(f'decomp_test/rho_uv_{uv}_qp_{qp}.npy', 'wb') as f:
                np.save(f, (rho, unc, Su, un_proj, un_proj_unc, angles_init))

# +++ MAIN SCRIPT +++
if __name__ == '__main__':
    # +++ hyper parameters
    SAMP = (5, 1)
    NUM_STEP = 30

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
    sweep_qp_phi(m, -38, 0, NUM_STEP, SAMP, f'decomp_test/qp_phi_sweep_{NUM_STEP}.csv')
    df = m.output_data(f'decomp_test/qp_phi_raw_{NUM_STEP}.csv')
    # sweep_full_tomo([2, 43, 6])
    # df = m.output_data('decomp_test/uv_rho_sweep.csv')
    m.shutdown()
    # show_data('qp_phi_sweep.csv', 'qp_phi_sweep_nofit.png')
    
    
    # sweep_qp_phi(None, SAMP, NUM_STEP, 'qp_phi_sweep_no_discontinuities.csv')
    # sweep_uvhwp_alph(None, SAMP, NUM_STEP, 'uvhwp_alph_sweep.csv')
    # sweep_bchwp_beta(None, SAMP, NUM_STEP, 'bchwp_beta_sweep.csv')
    

    # show_data('uvhwp_alpha_sweep.csv')


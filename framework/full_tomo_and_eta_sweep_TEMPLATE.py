from lab_framework import Manager
import numpy as np
import scipy.optimize as opt
from full_tomo_updated_richard import get_rho
from analysis_old import *
from rho_methods import get_fidelity, get_purity
import pandas as pd

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho(chi):

    H = ket([1,0])
    V = ket([0,1])
    D = ket([np.sqrt(0.5), np.sqrt(0.5)])
    A = ket([np.sqrt(0.5), -np.sqrt(0.5)])
    R = ket([np.sqrt(0.5), -1j * np.sqrt(0.5)])
    L = ket([np.sqrt(0.5), 1j * np.sqrt(0.5)])

    phi = (np.cos(chi/2) * np.kron(H, R) + np.exp(-1j*np.pi/6) * np.sin(chi/2) * np.kron(V, L))/np.sqrt(2) #TODO: change to current state

    rho = phi @ phi.conj().T

    return rho

if __name__ == '__main__':
    #TODO: Change these
    basisName = "hr_negpi_6_vl"
    mpName = "ria"
    date = "05232025"
    TRIAL = 1

    SWEEP_PARAMS = [-35, -1, 20, 5, 2]
    CHI_PARAMS = [0.001, np.pi/2, 6]

    # initialize the manager
    m = Manager('config.json')

    # make phi plus 
    m.make_state('phi_plus')
    # check count rates
    m.log('Checking HH and VV count rates...')
    m.meas_basis('HH')
    hh_counts = m.take_data(5,3,'C4')
    m.meas_basis('VV')
    vv_counts = m.take_data(5,3,'C4')

    # tell the user what is up
    print(f'HH count rates: {hh_counts}\nVV count rates: {vv_counts}')

    # check if the count rates are good
    inp = input('Continue? [y/n] ')
    if inp.lower() != 'y':
        print('Exiting...')
        m.shutdown()
        quit()

    # manually perform sweep of UVHWP for various chi values (chi/2=eta)
    chi_vals = np.linspace(*CHI_PARAMS)

    # Sweep UVHWP
    GUESS = -66 #TODO: Change if necessary for correct sign
    RANGE = 22.5
    N = 35
    SAMP = (5, 3)

    #TODO: CHANGE THIS to correct state
    m.make_state("phi_plus")
    m.B_C_HWP.goto(67.5)
    m.B_C_QWP.goto(45)
    m.C_QP.goto(-21.288)

    UVHWP_PARAMS = [GUESS - RANGE, GUESS + RANGE, N, *SAMP]

    #TODO: Change measurment bases to be correct for your state
    # configure measurement basis
    print(m.time, f'Configuring measurement basis HR')
    m.meas_basis('HR')

    # do sweep
    print(m.time, f'Beginning sweep of uvhwp from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: HR')
    m.sweep('C_UV_HWP', GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    # obtain the first round of data and switch to a new output file
    df1 = m.output_data(f"{mpName}_{basisName}/UVHWP_balance_sweep1.csv")
    data1 = pd.read_csv(f"{mpName}_{basisName}/UVHWP_balance_sweep1.csv")

    #TODO: Change measurement basis to correct one for your state
    # sweep in the second basis
    print(m.time, f'Configuring measurement basis VL')
    m.meas_basis('VL')

    print(m.time, f'Beginning sweep of uv_hwp from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: VL')
    m.sweep('C_UV_HWP', GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    print(m.time, 'Data collected')
    df2 = m.output_data(f'{mpName}_{basisName}/UVHWP_balance_sweep2.csv')
    data2 = pd.read_csv(f'{mpName}_{basisName}/UVHWP_balance_sweep2.csv')

    args1, unc1 = fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_SEM)
    args2, unc2 = fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_SEM)

    for chi in chi_vals:
        # Calculate the UVHWP angle we want for a given CHI value
        desired_ratio = (np.cos(chi/2) / np.sin(chi/2))**2
        def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple):
            ''' Function want to minimize'''
            return (sin2_sq(x_, *args1_) / sin2_sq(x_, *args2_) - desired_ratio)**2
        x_min, x_max = np.min(data1.C_UV_HWP), np.max(data1.C_UV_HWP)
        UVHWP_angle = opt.brute(min_me, args=(args1, args2), ranges=((x_min, x_max),))

        # might need to retune this if there are multiple roots. I'm only assuming one root
        m.C_UV_HWP.goto(UVHWP_angle)
        
        # measuring!
        rho, unc, Su, un_proj, un_proj_unc = get_rho(m, SAMP)

        actual_rho = get_theo_rho(chi)

        # print results
        print('measured rho\n---')
        print(rho)
        print('uncertainty \n---')
        print(unc)
        print('actual rho\n ---')
        print(actual_rho)

        # compute fidelity
        fidelity = get_fidelity(rho, actual_rho)
        print('fidelity', fidelity)
        purity = get_purity(rho)
        print('purity', purity)
        
        angles = [UVHWP_angle, -21.288, 67.5, 45] #TODO: change output data function to inlude B_C_QWP
        chi_save = np.rad2deg(chi) #naming convention (for it to work in process_expt) is in deg
        # save results
        with open(f"{mpName}_{basisName}/rho_({basisName}-{chi_save}-{TRIAL}).npy", 'wb') as f:
            np.save(f, (rho, unc, Su, un_proj, un_proj_unc, chi, angles, fidelity, purity))
        tomo_df = m.output_data(f'{mpName}_{basisName}/tomo_data_{basisName}_{chi_save}_{date}_{TRIAL}.csv')
    
    m.shutdown()
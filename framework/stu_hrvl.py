from lab_framework import Manager
import numpy as np
import scipy.optimize as opt
from full_tomo_updated_richard import get_rho
from analysis_old import *
from rho_methods import get_fidelity, get_purity
import pandas as pd


#### NOT COMPLETE ###


def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho(chi):

    H = ket([1,0])
    V = ket([0,1])
    D = ket([np.sqrt(0.5), np.sqrt(0.5)])
    A = ket([np.sqrt(0.5), -np.sqrt(0.5)])
    R = ket([np.sqrt(0.5), 1j * np.sqrt(0.5)])
    L = ket([np.sqrt(0.5), -1j * np.sqrt(0.5)])

    phi = (np.cos(chi/2) * np.kron(H, R) - 1j * np.sin(chi/2) * np.kron(V, L))/np.sqrt(2)

    rho = phi @ phi.conj().T

    return rho

if __name__ == '__main__':
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

    # setup the phase sweep
    m.reset_output()
    #x_vals = np.linspace(*SWEEP_PARAMS[:3])
    m.meas_basis('DA')
    m.sweep("C_QP", -35, -1, 20, 5, 3) #Sometimes the minimum is near the edge of the bounds in which case you won't get a parabola/normal angle. 

    print(m.time, "Sweep complete")

    # read the data into a dataframe
    df = m.output_data(f"stu_hrvl/QP_sweep.csv")
    data = pd.read_csv(f"stu_hrvl/QP_sweep.csv")

    # take the counts of the quartz sweep at each angle and find the index of the minimum data point
    QP_counts = data["C4"]
    print('QP Counts', QP_counts)
    min_ind = 0
    for i in range(len(QP_counts)):
        if QP_counts[i] == min(QP_counts):
            min_ind = i
    
    # creates a new data set of counts centered around the previous sweep's minimum data point
    new_guess = data["C_QP"][min_ind]
    RANGE = 5

    # finds the new minimum and maximum indices of the truncated data set
    min_bound = 0
    max_bound = len(QP_counts)

    # sets the minimum and maximum indices to the data point with the angle value (in degrees) new_guess +- RANGE
    for i in range(len(QP_counts)):
        if data["C_QP"][i] <= new_guess - RANGE:
            min_bound = i
        if data["C_QP"][i] >= new_guess + RANGE:
            max_bound = i
            break
    
    # create new truncated data set using min_bound and max_bound
    fit_data = QP_counts[min_bound:max_bound]
    fit_angles = data["C_QP"][min_bound:max_bound]
    fit_unc = data["C4_SEM"][min_bound:max_bound]
    print('fit data:', fit_data)
    print('fit angles:', fit_angles)
    print('fit unc:', fit_unc)
    # fits the truncated data set to a quartic fit function
    args1, unc1 = fit('quartic', fit_angles, fit_data, fit_unc)

    # finds the angle that corresponds to the minimum value of the fit function
    def fit_func(x):

        return args1[0] * x**4 + args1[1] * x**3 + args1[2] * x**2 + args1[3] * x + args1[4]

    # finds the angle at which the minimum of the fit function occurs to return as the QP angle setting
    minimum = opt.minimize(fit_func, new_guess)
    C_QP_angle = minimum.pop('x')

    # prints and returns the angle
    print('QP minimum angle is:', C_QP_angle)

    # set the qp angle
    m.C_QP.goto(C_QP_angle)

    # manually perform sweep of UVHWP
    chi_vals = np.linspace(*CHI_PARAMS)
    for chi in chi_vals:
        ### UV HWP SECTION ###
        GUESS = -65.86833 # flip all the quartz plate minimum so it actually minimizes
        RANGE = 22.5
        N = 35
        SAMP = (5, 3)
        # sweeping over chi values, tuning ratio first 
        chi_vals = np.linspace(*CHI_PARAMS)

        UVHWP_PARAMS = [GUESS - RANGE, GUESS + RANGE, N, *SAMP]

        # configure measurement basis
        print(m.time, f'Configuring measurement basis HH')
        m.meas_basis('HH')

        # do sweep
        print(m.time, f'Beginning sweep of uvhwp from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: HH')
        m.sweep('C_UV_HWP', GUESS-RANGE, GUESS+RANGE, N, *SAMP)

        # obtain the first round of data and switch to a new output file
        df1 = m.output_data(f"stu_hrvl/UVHWP_balance_sweep1.csv")
        data1 = pd.read_csv(f"stu_hrvl/UVHWP_balance_sweep1.csv")

        # sweep in the second basis
        print(m.time, f'Configuring measurement basis VV')
        m.meas_basis('VV')

        print(m.time, f'Beginning sweep of uv_hwp from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: VV')
        m.sweep('C_UV_HWP', GUESS-RANGE, GUESS+RANGE, N, *SAMP)

        print(m.time, 'Data collected')
        df2 = m.output_data(f'stu_hrvl/UVHWP_balance_sweep2.csv')
        data2 = pd.read_csv(f'stu_hrvl/UVHWP_balance_sweep2.csv')

        args1, unc1 = fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_SEM)
        args2, unc2 = fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_SEM)

        # Calculate the UVHWP angle we want.
        desired_ratio = (np.cos(chi/2) / np.sin(chi/2))**2
        def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple):
            ''' Function want to minimize'''
            return (sin2_sq(x_, *args1_) / sin2_sq(x_, *args2_) - desired_ratio)**2
        x_min, x_max = np.min(data1.C_UV_HWP), np.max(data1.C_UV_HWP)
        UVHWP_angle = opt.brute(min_me, args=(args1, args2), ranges=((x_min, x_max),))

        # find the fit function and optimal value 
        # coefficients = np.polyfit(angles, ratios, 5)
        # a, b, c, d, e = coefficients
        # e = e - desired_ratio
        # UVHWP_angle = np.roots(coefficients)[0]

        # might need to retune this if there are multiple roots. I'm only assuming one root
        m.configure_motors(C_UV_HWP = UVHWP_angle, 
                           B_C_HWP = 67.5,
                           B_C_QWP = 90)
        
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
        
        # 67.5 -> B_C_HWP, 90 -> B_C_QWP
        angles = [UVHWP_angle, C_QP_angle, 67.5, 90] # change output data function to inlude B_C_QWP

        # save results
        with open(f"stu_hrvl/rho_('E0', (45.0, {chi}))_1.npy", 'wb') as f:
            np.save(f, (rho, unc, Su, un_proj, un_proj_unc, chi, angles, fidelity, purity))
        date = "7022024"
        tomo_df = m.output_data(f'stu_hrvl/tomo_data_{chi}_{date}.csv')
    
    m.shutdown()
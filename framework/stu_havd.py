from lab_framework import Manager
import numpy as np
import scipy.optimize as opt
from full_tomo_updated_richard import get_rho
from analysis_old import *
from rho_methods import get_fidelity, get_purity
import pandas as pd

"""
TODO:   
        fix the B_C_HWP and B_C_QWP angles
        make sure QP is set to right angle
        make sure B_C_HWP and B_C_QWP are set to right angles 
"""

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho(chi):

    H = ket([1,0])
    V = ket([0,1])
    D = ket([np.sqrt(0.5), np.sqrt(0.5)])
    A = ket([np.sqrt(0.5), -np.sqrt(0.5)])
    R = ket([np.sqrt(0.5), 1j * np.sqrt(0.5)])
    L = ket([np.sqrt(0.5), -1j * np.sqrt(0.5)])

    phi = (np.cos(chi/2) * np.kron(H, A) - 1j * np.sin(chi/2) * np.kron(V, D))

    rho = phi @ phi.conj().T

    return rho

if __name__ == '__main__':
    SWEEP_PARAMS = [-35, -1, 20, 5, 2]
    CHI_PARAMS = [0.001, np.pi/2, 6]

    # initialize the manager
    m = Manager('config.json')


    # Manually found QP angle to be -24.8.
    # # make phi plus 
    # m.make_state('phi_plus') 
    # #EXPT RHO: [[ 0.47589049+0.j         -0.06945163-0.08411542j  0.07464098-0.07892626j 0.44664475+0.03311149j]
    # #[-0.06945163+0.08411542j  0.02195016+0.j          0.00581658+0.03551439j -0.06989104+0.08194604j]
    # #[ 0.07464098+0.07892626j  0.00581658-0.03551439j  0.03664963+0.j 0.06780289+0.08958259j]
    # #[ 0.44664475-0.03311149j -0.06989104-0.08194604j  0.06780289-0.08958259j 0.46550973+0.j        ]]
    # m.log('Checking HH and VV count rates...')
    # m.meas_basis('HH')
    # hh_counts = m.take_data(5,3,'C4')
    # m.meas_basis('VV')
    # vv_counts = m.take_data(5,3,'C4')

    # # tell the user what is up
    # print(f'HH count rates: {hh_counts}\nVV count rates: {vv_counts}')

    # # check if the count rates are good
    # inp = input('Continue? [y/n] ')
    # if inp.lower() != 'y':
    #     print('Exiting...')
    #     m.shutdown()
    #     quit()

    # # setup the phase sweep
    # m.reset_output()
    # #x_vals = np.linspace(*SWEEP_PARAMS[:3])
    # m.meas_basis('DL')
    # m.configure_motors(C_UV_HWP =-112.2352648283306,
    #                    B_C_HWP = 0,
    #                    B_C_QWP = 0)
    # m.sweep("C_QP", -35, -1, 20, 5, 3) #Sometimes the minimum is near the edge of the bounds in which case you won't get a parabola/normal angle. 
    # print(m.time, "Sweep complete")

    # # read the data into a dataframe
    # df = m.output_data(f"stu_havd_trial_4/QP_sweep.csv")
    # data = pd.read_csv(f"stu_havd_trial_4/QP_sweep.csv")

    # # take the counts of the quartz sweep at each angle and find the index of the minimum data point
    # QP_counts = data["C4"]
    # print('QP Counts', QP_counts)
    # min_ind = 0
    # for i in range(len(QP_counts)):
    #     if QP_counts[i] == min(QP_counts):
    #         min_ind = i
    
    # # creates a new data set of counts centered around the previous sweep's minimum data point
    # new_guess = data["C_QP"][min_ind]
    # RANGE = 5

    # # finds the new minimum and maximum indices of the truncated data set
    # min_bound = 0
    # max_bound = len(QP_counts)

    # # sets the minimum and maximum indices to the data point with the angle value (in degrees) new_guess +- RANGE
    # for i in range(len(QP_counts)):
    #     if data["C_QP"][i] <= new_guess - RANGE:
    #         min_bound = i
    #     if data["C_QP"][i] >= new_guess + RANGE:
    #         max_bound = i
    #         break
    
    # # create new truncated data set using min_bound and max_bound
    # fit_data = QP_counts[min_bound:max_bound]
    # fit_angles = data["C_QP"][min_bound:max_bound]
    # fit_unc = data["C4_SEM"][min_bound:max_bound]
    # print('fit data:', fit_data)
    # print('fit angles:', fit_angles)
    # print('fit unc:', fit_unc)
    # # fits the truncated data set to a quartic fit function
    # args1, unc1 = fit('quartic', fit_angles, fit_data, fit_unc)

    # # finds the angle that corresponds to the minimum value of the fit function
    # def fit_func(x):

    #     return args1[0] * x**4 + args1[1] * x**3 + args1[2] * x**2 + args1[3] * x + args1[4]

    # # finds the angle at which the minimum of the fit function occurs to return as the QP angle setting
    # minimum = opt.minimize(fit_func, new_guess)
    # C_QP_angle = minimum.pop('x')

    # # prints and returns the angle
    # print('QP minimum angle is:', C_QP_angle)

    # set the qp angle
    m.C_QP.goto(-24.8)
    m.C_PCC.goto(-0.3063616071428328)
    # manually perform sweep of UVHWP
    chi_vals = np.linspace(*CHI_PARAMS)
    for chi in chi_vals:
        ### UV HWP SECTION ###
        GUESS = -112.2352648283306 # flip all the quartz plate minimum so it actually minimizes
        RANGE = 22.5
        N = 30
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
        df1 = m.output_data(f"stu_havd_trial_6/UVHWP_balance_sweep1.csv")
        data1 = pd.read_csv(f"stu_havd_trial_6/UVHWP_balance_sweep1.csv")

        # sweep in the second basis
        print(m.time, f'Configuring measurement basis VV')
        m.meas_basis('VV')

        print(m.time, f'Beginning sweep of uv_hwp from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: VV')
        m.sweep('C_UV_HWP', GUESS-RANGE, GUESS+RANGE, N, *SAMP)

        print(m.time, 'Data collected')
        df2 = m.output_data(f'stu_havd_trial_6/UVHWP_balance_sweep2.csv')
        data2 = pd.read_csv(f'stu_havd_trial_6/UVHWP_balance_sweep2.csv')

        args1, unc1 = fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_SEM)
        args2, unc2 = fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_SEM)

        # Calculate the UVHWP angle we want.
        desired_ratio = (np.cos(chi/2) / np.sin(chi/2))**2
        def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple):
            ''' Function want to minimize'''
            return (sin2_sq(x_, *args1_) / sin2_sq(x_, *args2_) - desired_ratio)**2
        x_min, x_max = np.min(data1.C_UV_HWP), np.max(data1.C_UV_HWP)
        UVHWP_angle = opt.brute(min_me, args=(args1, args2), ranges=((x_min, x_max),),finish=None)

        print('UVHWP angle : ', UVHWP_angle)

        # might need to retune this if there are multiple roots. I'm only assuming one root
        m.configure_motors(C_UV_HWP = UVHWP_angle, 
                           B_C_HWP = 112.5,
                           B_C_QWP = 135)
        
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
        angles = [UVHWP_angle, -24.8, 112.5, 135] # change output data function to inlude B_C_QWP
        chi_save = np.rad2deg(chi) #naming convention (for it to work in process_expt) is in deg
        # save results
        with open(f"stu_havd_trial_6/rho_('E0', (45.0, {chi_save}))_1.npy", 'wb') as f:
            np.save(f, (rho, unc, Su, un_proj, un_proj_unc, chi, angles, fidelity, purity))
        date = "7152024"
        tomo_df = m.output_data(f'stu_havd_trial_6/tomo_data_{chi_save}_{date}.csv')
    
    m.shutdown()
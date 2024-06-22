from lab_framework import Manager
import numpy as np
import scipy.optimize as opt
from full_tomo_updated_richard import get_rho
from analysis_old import *
from rho_methods import get_fidelity, get_purity

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

    # PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    # PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)
    # phi = np.cos(alpha)*PHI_PLUS + np.exp(1j*beta)*np.sin(alpha)*PHI_MINUS

    rho = phi @ phi.conj().T

    return rho

if __name__ == '__main__':
    SWEEP_PARAMS = [-10, 10, 20, 5, 2]
    CHI_PARAMS = [0, np.pi, 6]

    # initialize the manager
    m = Manager('../config.json')

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
    x_vals = np.linspace(*SWEEP_PARAMS[:3])

    # Quartz plate sweep
    m.log("Performing QP sweep")

    # we expect minimum counts for this basis
    m.meas_basis('DR')

    # sweep the quartz plate
    _, data = m.sweep('C_QP', *SWEEP_PARAMS) 

    # fit a quadratic function (5th degree polynomial)
    coefficients = np.polyfit(x_vals, data, 5)
    a, b, c, d, e = coefficients

    def fit_func(x):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    # find the minimum of this fit
    initial_guess = 0
    C_QP_angle = opt.minimize(fit_func, initial_guess)
    print('Minimum  QP angle: ', C_QP_angle)

    
    # sweeping over chi values, tuning ratio first 
    chi_vals = np.linspace(*CHI_PARAMS)

    UVHWP_PARAMS = []
    SAMP = (5, 1)

    # manually perform sweep of UVHWP
    for chi in chi_vals:
        # calculate the ratio
        desired_ratio = np.cos(chi/2)**2 + np.sin(chi/2)**2

        # run the sweep manually
        angles = np.linspace(UVHWP_PARAMS[:3])
        for angle in angles:
            m.configure_motors(C_UV_HWP = angle)
            m.meas_basis('HH')
            hh_counts = m.take_data(5, 3,'C4')
            m.meas_basis('VV')
            vv_counts = m.take_data(5, 3,'C4')

        ratios = hh_counts / vv_counts

        # find the fit function and optimal value 
        coefficients = np.polyfit(angles, ratios, 5)
        a, b, c, d, e = coefficients
        e = e - desired_ratio
        UVHWP_angle = np.roots(coefficients)[0]

        # might need to retune this if there are multiple roots. I'm only assuming one root
        m.configure_motors(C_UV_HWP = UVHWP_angle, 
                           B_C_QWP = np.pi/4)
        
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
        
        # 0 -> B_C_HWP, 45 -> B_C_QWP
        angles = [UVHWP_angle, C_QP_angle, 0, 45] # change output data function to inlude B_C_QWP

        # save results
        with open(f"int_state_sweep_WP3_2/rho_('E0', {chi})_1.npy", 'wb') as f:
            np.save(f, (rho, unc, Su, un_proj, un_proj_unc, chi, angles, fidelity, purity))
        date = "6032024"
        tomo_df = m.output_data(f'int_state_sweep_WP3_2/tomo_data_{chi}_{date}.csv')
    
    m.shutdown()
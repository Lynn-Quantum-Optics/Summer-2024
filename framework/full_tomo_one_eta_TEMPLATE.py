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
    chi = 90.0 #TODO: change to your value

    #TODO: Change these
    basisName = "hr_negpi_6_vl"
    mpName = "ria"
    date = "05232025"
    TRIAL = 1

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


    #TODO: CHANGE THIS to correct state
    m.make_state("phi_plus")
    m.B_C_HWP.goto(67.5)
    m.B_C_QWP.goto(45)
    m.C_QP.goto(-21.288)
    m.C_UV_HWP.goto(-66)
    
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
    
    angles = [-66 -21.288, 67.5, 45] #TODO: change to your values
    chi_save = np.rad2deg(chi) #naming convention (for it to work in process_expt) is in deg
    # save results
    with open(f"{mpName}_{basisName}/rho_({basisName}-{chi_save}-{TRIAL}).npy", 'wb') as f:
        np.save(f, (rho, unc, Su, un_proj, un_proj_unc, chi, angles, fidelity, purity))
    tomo_df = m.output_data(f'{mpName}_{basisName}/tomo_data_{basisName}_{chi_save}_{date}_{TRIAL}.csv')
    
    m.shutdown()
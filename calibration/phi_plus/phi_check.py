from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp

# Purpose of program is to be run pre-pcc_sweep to ensure the values of the Phi Plus state are good.
if __name__ == '__main__':
    # initialize the manager
    m = Manager('../config.json')
    # setup the phi plus state using values from ratio_tuning and phasing_finding 
    m.C_UV_HWP.goto(65.39980)
    m.C_QP.goto(-24.1215)
    m.C_PCC.goto(135) # ballpark based on an old calibration
    m.B_C_HWP.goto(0)
    
    # check count rates, expected HH>>HV, DD>>DA, RR<<RL.
    # m.log('Checking HH and VV count rates...')
    # m.meas_basis('HH')
    # hh_counts = m.take_data(5,3,'C4')
    # m.meas_basis('HV')
    # hv_counts = m.take_data(5,3,'C4')
    # m.meas_basis('DD')
    # dd_counts = m.take_data(5,3,'C4')
    # m.meas_basis('DA')
    # da_counts = m.take_data(5,3,'C4')
    # m.meas_basis('RR')
    # rr_counts = m.take_data(5,3,'C4')
    # m.meas_basis('RL')
    # rl_counts = m.take_data(5,3,'C4')

    m.log('Performing sweeps. This will take a while.')

    # go through the bases and sweep each one across the angles
    for basis in ['HH', 'VV', 'HV', 'VH', 'DD', 'DD', 'DA', 'AD', 'RR', 'LL', 'RL', 'LR']:
        m.log(f'Beginning {basis} sweep...')
        # setup the measurement basis
        # basis_of_loop = f"{basis}_counts"
        m.meas_basis(basis)
        #globals()[basis_of_loop] 
        counts = m.take_data(5, 3, 'C4')
        print(counts)
    m.shutdown()

    # tell the user what occured
    # print(f'HH count rates: {hh_counts}\nHV count rates: {hv_counts}')
    # print(f'DD count rates: {dd_counts}\nDA count rates: {da_counts}')
    # print(f'RR count rates: {rr_counts}\nRL count rates: {rl_counts}')
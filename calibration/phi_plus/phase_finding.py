from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp

# Approx 25 mins.

if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    SWEEP_PARAMS = [-35, -15, 10, 3, 1]

    # initialize the manager
    m = Manager('../config.json')

    # setup the superposition state - this is to make a state from scratch
    #m.C_UV_HWP.goto(22.5) # based on the ratio finding  #65.38956684313322
    #m.C_QP.goto(25.8119041041324) #-25.8119041041324
    #m.C_PCC.goto(2.894736842105263) # ballpark based on an old calibration
    #m.B_C_HWP.goto(0)

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

    # delete this later, only used for HDVA state phase finding
    m.B_C_HWP.goto(67.5)
    m.B_C_QWP.goto(45)
    
    # setup the phase sweep
    m.reset_output()
    datas = {'QP': np.linspace(*SWEEP_PARAMS[:3])}

    m.log('Performing sweeps. This will take a while.')
    # go through the bases and sweep each one across the angles
    for basis in ['DR', 'DL', 'AR', 'AL', 'RR', 'RL', 'LR', 'LL']:
        m.log(f'Beginning {basis} sweep...')
        # setup the measurement basis
        m.meas_basis(basis)
        # sweep the quartz plate
        _, datas[basis] = m.sweep('C_QP', *SWEEP_PARAMS)
        # output the data for this sweep
        m.output_data(f'QP_{basis}_sweep')
    
    m.shutdown()

    # save the overall data
    print('Saving all sweep data...')
    pd.DataFrame(datas).to_csv('all_phase_finding_data2_test05202025.csv')

    # calculate the phase difference
    datas['phi'] = unp.arctan2((datas['DR'] - datas['DL'] - datas['AR'] + datas['AL']),(-datas['RR'] + datas['RL'] + datas['LR'] - datas['LL']))

    # save the data
    pd.DataFrame(datas).to_csv('phi_data_05202025.csv')

    



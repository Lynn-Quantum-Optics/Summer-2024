from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp

if __name__ == '__main__':
    SWEEP_PARAMS = [-8, 8, 20, 5, 1]

    # initialize the manager
    m = Manager('../config.json')

    # setup the superposition state
    m.C_UV_HWP.goto(22.5)
    m.C_QP.goto(0)
    m.C_PCC.goto(4.005) # ballpark based on an old calibration
    m.B_C_HWP.goto(0)
    
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
    pd.DataFrame(datas).to_csv('all_sweep_data.csv')

    # calculate the phase difference
    datas['phi'] = unp.arctan2((datas['DR'] - datas['DL'] - datas['AR'] + datas['AL']),(-datas['RR'] + datas['RL'] + datas['LR'] - datas['LL']))

    # save the data
    pd.DataFrame(datas).to_csv('phi_data.csv')

    



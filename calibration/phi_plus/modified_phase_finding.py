from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp

# Approx 25 mins.

if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    qp_loc = [-20. -19, -18, 17, -16, -15, -14, -13, -12, -11, -10]
    SWEEP_PARAMS = [-20, -10, 10, 5, 3]

    # initialize the manager
    m = Manager('../config.json')

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
    datas = {'QP': np.linspace(*SWEEP_PARAMS[:3])}

    m.log('Performing sweeps. This will take a while.')
    # go through the bases and sweep each one across the angles
    for basis in ['DR', 'DL', 'AR', 'AL', 'RR', 'RL', 'LR', 'LL']:
        for i in range(len(qp_loc)):
            angle = qp_loc[i]
            SWEEP_PARAMS[0] = angle
            SWEEP_PARAMS[1] = angle+1
            m.log(f'Beginning {angle} sweep...')
            # setup the measurement basis
            m.meas_basis(basis)
            # sweep the quartz plate
            _, datas[basis] = m.sweep('C_QP', *SWEEP_PARAMS)
            # output the data for this sweep
            m.output_data(f'QP_{angle}_{basis}_sweep')
    
    m.shutdown()

    # save the overall data
    print('Saving all sweep data...')
    pd.DataFrame(datas).to_csv('mod_all_phase_finding_data2_test05292025.csv')

    # calculate the phase difference
    datas['phi'] = unp.arctan2((datas['DR'] - datas['DL'] - datas['AR'] + datas['AL']),(-datas['RR'] + datas['RL'] + datas['LR'] - datas['LL']))

    # save the data
    pd.DataFrame(datas).to_csv('mod_phi_data_05292025.csv')

    



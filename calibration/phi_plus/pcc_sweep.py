from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp

if __name__ == '__main__':
    #SWEEP_PARAMS = [-7, 8, 30, 5, 3]
    #SWEEP_PARAMS = [-15, 15, 30, 5, 3]
    SWEEP_PARAMS = [0, 2, 1, 10, 6] # use for checking state purity at the PCC angle of 1 degree (current place for phi_plus)
    # initialize the manager
    m = Manager('../config.json')

    # setup the superposition state, assuming phi_plus has been updated in config
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
    datas = {'PCC': np.linspace(*SWEEP_PARAMS[:3])}

    m.log('Performing sweeps. This will take a while.')
    # go through the bases and sweep each one across the angles
    for basis in ['DD', 'AA', 'DA', 'AD']:
        m.log(f'Beginning {basis} sweep...')
        # setup the measurement basis
        m.meas_basis(basis)
        # sweep the quartz plate
        _, datas[basis] = m.sweep('C_PCC', *SWEEP_PARAMS)
        # output the data for this sweep
        m.output_data(f'PCC_{basis}_sweep')
    
    m.shutdown()

    # save the overall data
    print('Saving all sweep data...')
    pd.DataFrame(datas).to_csv('pcc_sweep_test_05222025_2.csv')

    # calculate the purity of the state
    datas['purity'] = (datas['DD'] + datas['AA'] - (datas['DA'] + datas['AD']))/(datas['DD'] + datas['AA'] + datas['DA'] + datas['AD'])

    # save the data
    pd.DataFrame(datas).to_csv('purity_test_05222025_2.csv')

    



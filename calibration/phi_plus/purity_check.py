from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp

if __name__ == '__main__':
    # initialize the manager
    m = Manager('../config.json')

    # setup the superposition state, assuming phi_plus has been updated in config
    m.make_state('phi_plus')

    datas = np.linspace(0, 4, 4)
    b = 0
    m.log('Performing sweeps. This will take a while.')

    # go through the bases and sweep each one across the angles
    for basis in ['DD', 'AA', 'DA', 'AD']:
        m.log(f'Beginning {basis} measurement...')
        # setup the measurement basis
        m.meas_basis(basis)
        datas[b] = m.take_data(10, 5, 'C4')
        # output the data for this sweep
        m.output_data(f'PCC_{basis}_sweep')
        b += 1
    
    m.shutdown()

    # save the overall data
    print('Saving all sweep data...')
    pd.DataFrame(datas).to_csv('purity_test_05212025.csv')

    # calculate the purity of the state
    datas['purity'] = (datas[0] + datas[1] - (datas[2] + datas[3]))/(datas[0] + datas[1] + datas[2] + datas[3])

    # save the data
    pd.DataFrame(datas).to_csv('purity_test_05212025.csv')

    

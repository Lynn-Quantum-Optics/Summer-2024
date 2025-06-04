from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

# this file determines the angles theta and gamma of a given state sin(theta)|H>|alpha>+(e^igamma)cos(theta)|V>|alpha_perp>
# this file is used to check a state that

if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    PARAMS = [5, 3]
    DATE = "06022025" #please update
    TRIAL = 3 #please update
    STATE = "hd_negpi_3_va"

    fileName = f"gamma_determination_{DATE}_{TRIAL}_for_{STATE}"

    # initialize the manager
    m = Manager('../config.json')

    # parameters used for HR + e^-ipi/6 VL sstate
    m.log('Setting up state')
    m.make_state('phi_plus')
    m.B_C_HWP.goto(67.5)
    m.B_C_QWP.goto(45)
    m.C_QP.goto(-27.027) # change
    m.C_UV_HWP.goto(-115.2803939016242)


    datas = {'QP': np.linspace(1, 2, 1, endpoint=False)}

    for basis in ['HD', 'VA', 'RA', 'RD', 'LA', 'LR']:
        m.log(f'Beginning {basis} measurement...')
        # setup the measurement basis
        m.meas_basis(basis)
        # measuring
        _, datas[basis] = m.take_data('C_QP', *PARAMS)
        m.output_data(f'QP_{basis}_measurement')
    
    m.shutdown()

    # save the overall data
    print('Saving all sweep data...')
    pd.DataFrame(datas).to_csv(f'raw_gamma_data_{DATE}_{TRIAL}_for_{STATE}.csv')

    datas['theta'] = unp.arctan2(datas['HD']/datas['VA'])
    datas['gamma1'] = unp.rad2deg(-1*unp.arcsin((datas['RD']+datas['LA']-datas['RA']-datas['LD'])/(2*np.cos(datas['theta'])*np.sin(datas['theta'])))) # antidiagonal
    datas['gamma2'] = unp.rad2deg(-1*unp.arcsin((datas['RD']+datas['RA']-datas['LD']-datas['LA'])/(2*np.cos(datas['theta'])*np.sin(datas['theta']))))  # diagonal

    datas['theta'] = unp.rad2deg(datas['theta'])

    pd.DataFrame(datas).to_csv(f'{fileName}.csv')
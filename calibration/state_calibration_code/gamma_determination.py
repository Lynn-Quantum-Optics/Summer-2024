from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

# this file determines the angle gamma of a given state sin(theta)|H>|alpha>+(e^igamma)cos(theta)|V>|alpha_perp>
# this file is used to calibrate the qp angle

if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    QP_angle = -33
    DATE = "06052025" #please update
    STATE = "hd_negpi_3_va"
    
    fileName = f"gamma_determination_{DATE}_{QP_angle}_for_{STATE}"
    PARAMS = [QP_angle, QP_angle+1, 1, 5, 3]

    # initialize the manager
    m = Manager('../config.json')

    # parameters used for HR + e^-ipi/6 VL sstate
    m.log('Setting up state')
    m.make_state('phi_plus')
    m.B_C_HWP.goto(67.5)
    m.B_C_QWP.goto(45)
    m.C_QP.goto(QP_angle) # change
    m.C_UV_HWP.goto(-115.2803939016242)


    datas = {'QP': np.linspace(QP_angle, PARAMS[1], 1, endpoint=False)}

    for basis in ['RA', 'RD', 'LA', 'LR', 'RR', 'LL', 'RL', 'LR']:
        m.log(f'Beginning {basis} measurement...')
        # setup the measurement basis
        m.meas_basis(basis)
        # measuring
        m.log(f'Measuring {basis}')
        _, datas[basis] = m.sweep('C_QP', *PARAMS)
        m.output_data(f'QP_{basis}_measurement')
    
    m.shutdown()

    # save the overall data
    print('Saving all sweep data...')
    pd.DataFrame(datas).to_csv(f'raw_gamma_data_{DATE}_{QP_angle}_for_{STATE}.csv')

    datas['gamma'] = unp.arctan2((datas['RD']+datas['lA']-datas['LD']-datas['rA']), (datas['RR']+datas['LL']-datas['LR']-datas['RL']))
    datas['gammaDeg'] = unp.rad2deg(datas['gamma'])
    pd.DataFrame(datas).to_csv(f'{fileName}.csv')
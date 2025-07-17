from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

# this file determines the angle gamma of a given state sin(theta)|H>|alpha>+(e^igamma)cos(theta)|V>|alpha_perp>
# this file is used to calibrate the qp angle

if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    QP_angle = -25 #update as necessary...
    DATE = "07162025" #please update
    STATE = "ha_negpi_3_vd"
    TRIAL = 4
    
    fileName = f"QP_calibration/all_gamma_data_{DATE}_for_{STATE}_chi18_{TRIAL}"
    PARAMS = [QP_angle, QP_angle+5, 5, 5, 3]

    # initialize the manager
    m = Manager('../config.json')

    # parameters used for the state
    m.log('Setting up state')
    m.make_state('phi_plus')
    m.B_C_HWP.goto(22.5)
    m.B_C_QWP.goto(45)
    m.C_UV_HWP.goto(-130.44066667)

    # m.B_C_HWP.goto(0)
    # m.B_C_QWP.goto(-45)
    # m.C_UV_HWP.goto(-111.29939852262797)

    # m.B_C_HWP.goto(67.5)
    # m.B_C_QWP.goto(45)
    # m.C_UV_HWP.goto(-178.43334997)
    #m.C_UV_HWP.goto(-65.42683049252159) #-115.10003140098172


    datas = {'QP': np.linspace(QP_angle, PARAMS[1], PARAMS[2], endpoint=False)}

    #bases for hdva & havd state: 'DR', 'AR', 'DL', 'AL', 'RR', 'LL', 'RL', 'LR'
    #bases for hrvl state: 'DD', 'DA', 'AD', 'AA', 'RD', 'RA', 'LD', 'LA'

    for basis in ['DR', 'AR', 'DL', 'AL', 'RR', 'LL', 'RL', 'LR']:
        m.log(f'Beginning {basis} measurement...')
        # setup the measurement basis
        m.meas_basis(basis)
        # measuring
        m.log(f'Measuring {basis}')
        _, datas[basis] = m.sweep('C_QP', *PARAMS)
        m.output_data(f'QP_{basis}_measurement')

    # save the overall data
    #print('Saving all sweep data...')
    #pd.DataFrame(datas).to_csv(f'raw_gamma_data_{DATE}_{QP_angle}_for_{STATE}.csv')

    # for hdva state: 
    # datas['gamma'] = unp.arctan2(-(datas['DR']+datas['AL']-datas['DL']-datas['AR']), (datas['RR']+datas['LL']-datas['LR']-datas['RL']))

    # for havd
    datas['gamma'] = unp.arctan2((datas['DR']+datas['AL']-datas['DL']-datas['AR']), -(datas['RR']+datas['LL']-datas['LR']-datas['RL']))

    # for hrvl state:
    # datas['gamma'] = unp.arctan2((datas['DD']+datas['AA']-datas['DA']-datas['AD']), -(datas['RD']+datas['LA']-datas['RA']-datas['LD']))
    pd.DataFrame(datas).to_csv(f'{fileName}.csv')

    m.shutdown()
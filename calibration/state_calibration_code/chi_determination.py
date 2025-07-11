from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import pandas as pd

# calculates chi/2 for given UVHWP angle

if __name__ == '__main__':
    TRIAL = 1 #PLEASE UPDATE
    DATE = "07102025" #PLEASE UPDATE
    UVHWP_angles = [0] #[-42.90096457, -87.72854883]
    STATE = "phi_plus"
    fileName = f"chi_determination_{DATE}_for_{STATE}" 

    # initialize manager
    m = Manager('../config.json')

    m.make_state('phi_plus')

    # # parameters used for setting up the state
    # m.log('Setting up state')
    # m.B_C_HWP.goto(22.5)
    # m.B_C_QWP.goto(45)
    # m.C_QP.goto(-7.165756064967105)
    
    # m.B_C_HWP.goto(67.5)
    # m.B_C_QWP.goto(45)
    # m.C_QP.goto(-9.71449207491852) #-26.98906575253135

    # m.B_C_HWP.goto(0)
    # m.B_C_QWP.goto(-45)
    # m.C_QP.goto(-13)

    hh_counts = []
    vv_counts = []
    chis = []
    
    for angle in UVHWP_angles:
        PARAMS = [angle, angle+1, 1, 6, 5]
        m.log(f'Beginning {angle} measurement...')
        # setup the measurement basis
        m.log(f'Measuring HH')
        m.meas_basis('HH')
        _, hh = m.sweep('C_UV_HWP', *PARAMS)
        hh_counts.append(hh)
        
        m.log(f'Measuring VV')
        m.meas_basis('VV')
        _, vv = m.sweep('C_UV_HWP', *PARAMS)
        vv_counts.append(vv)

        chis.append(unp.arctan(unp.sqrt((vv)/hh)))

    datas = pd.DataFrame({'UVHWP': UVHWP_angles, 'HH' : hh_counts, 'VV' : vv_counts, 'Chi' : chis})
    
    pd.DataFrame(datas).to_csv(f'{fileName}.csv')
    
    m.shutdown()

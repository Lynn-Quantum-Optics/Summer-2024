from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import pandas as pd

# calculates chi for given UVHWP angle

if __name__ == '__main__':
    TRIAL = 1 #PLEASE UPDATE
    DATE = "07172025" #PLEASE UPDATE
    UVHWP_angles = [-131.29599484]
    QP_angles = [-21.986341777600742]
    chis_theo = np.linspace(0.001, np.pi/2, 6)[1:2]

    STATE = "ha_negpi_3_vd"
    fileName = f"chi_check_{DATE}_for_{STATE}_1" 

    # initialize manager
    m = Manager('../config.json')

    m.make_state('phi_plus')

    # parameters used for setting up the state
    m.log('Setting up state')
    m.B_C_HWP.goto(22.5)
    m.B_C_QWP.goto(45)

    ha_counts = []
    vd_counts = []
    chis = []
    counter = 0
    
    for angle in UVHWP_angles:
        m.C_QP.goto(QP_angles[counter])
        PARAMS = [angle, angle+1, 1, 6, 5]
        m.log(f'Beginning {angle} measurement...')
        # setup the measurement basis
        m.log(f'Measuring HA')
        m.meas_basis('HA')
        _, ha = m.sweep('C_UV_HWP', *PARAMS)
        ha_counts.append(ha)
        
        m.log(f'Measuring VD')
        m.meas_basis('VD')
        _, vd = m.sweep('C_UV_HWP', *PARAMS)
        vd_counts.append(vd)

        chis.append(2*unp.arctan(unp.sqrt((vd)/ha)))
        counter += 1

    datas = pd.DataFrame({'UVHWP': UVHWP_angles, 'QP' : QP_angles, 'HA' : ha_counts, 'VD' : vd_counts, 'Theo Chi' : chis_theo,'Chi' : chis})
    
    pd.DataFrame(datas).to_csv(f'{fileName}.csv')
    
    m.shutdown()

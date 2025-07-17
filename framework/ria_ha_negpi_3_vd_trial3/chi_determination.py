from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import pandas as pd

# calculates chi for given UVHWP angle

if __name__ == '__main__':
    TRIAL = 2 #PLEASE UPDATE
    DATE = "07142025" #PLEASE UPDATE
    UVHWP_angles = [-132.50543769, -127.38849844, -122.98117298, -118.51952741, -114.11387723]
    QP_angles = [-24.251088513826073, -25.20815734863281, -25.603982222707646, -25.831407406455597, -26.00327309056332]
    chis_theo = np.linspace(0.001, np.pi/2, 6)[1:]

    STATE = "ha_negpi_3_vd"
    fileName = f"chi_check2_{DATE}_for_{STATE}" 

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

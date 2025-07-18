from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp

if __name__ == '__main__':
    TRIAL = 1 #PLEASE UPDATE
    DATE = "07172025" #PLEASE UPDATE
    STATE = "ha_negpi_3_vd"
    fileName = f"ratio_tuning_{DATE}_for_{STATE}_chi18_{TRIAL}" 
    chis = np.linspace(0.001, np.pi/2, 6)
    chi = chis[1]
    SWEEP_PARAMETERS = [-118, -133, 15, 5, 3] #fine sweep

    # initialize manager
    m = Manager('../config.json')

    m.make_state('phi_plus')

    # # parameters used for setting up the state
    m.log('Setting up state')
    m.B_C_HWP.goto(22.5)
    m.B_C_QWP.goto(45)
    m.C_QP.goto(-22.757037032277957)

    # sweep UVHWP
    m.meas_basis('HA')
    _, ha_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'UVHWP_calibration/ha_sweep_{chi}_{TRIAL}.csv')
    m.meas_basis('VD')
    _, vd_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'UVHWP_calibration/vd_sweep_{chi}_{TRIAL}.csv')
    
    
    angles = np.linspace(*SWEEP_PARAMETERS[:3])
    thetas = unp.arctan(unp.sqrt((vd_rates)/(ha_rates)))
    params = analysis.fit('line', angles, thetas)
    analysis.plot_func('line', params, angles, color='blue')
    analysis.plot_errorbar(angles, thetas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Theta Parameter (rad)')
    plt.savefig(f'UVHWP_calibration/{fileName}_{TRIAL}.png', dpi=600)
    # plt.show()

    x = analysis.find_value('line', params, chi/2, angles)
    print(f'{chi} at {x}')

    m.shutdown()
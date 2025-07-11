from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp

if __name__ == '__main__':
    TRIAL = 2 #PLEASE UPDATE
    DATE = "07112025" #PLEASE UPDATE
    STATE = "ha_negpi_3_vd"
    fileName = f"ratio_tuning_{DATE}_for_{STATE}" 
    
    SWEEP_PARAMETERS = [-60, -75, 15, 5, 3] #fine sweep
    #SWEEP_PARAMETERS = [-55, -85, 15, 3, 1] #coarse sweep
    # SWEEP_PARAMETERS = [-40, -95, 40, 5, 3] #broad sweep

    # initialize manager
    m = Manager('../config.json')

    m.make_state('phi_plus')

    # # parameters used for setting up the state
    m.log('Setting up state')
    m.B_C_HWP.goto(22.5)
    m.B_C_QWP.goto(45)
    m.C_QP.goto(-3.8691303453947365)
    
    # m.B_C_HWP.goto(67.5)
    # m.B_C_QWP.goto(45)
    # m.C_QP.goto(-9.71449207491852) #-26.98906575253135

    # m.B_C_HWP.goto(0)
    # m.B_C_QWP.goto(-45)
    # m.C_QP.goto(-13)

    # sweep UVHWP
    m.meas_basis('HA')
    _, ha_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'ha_sweep_{TRIAL}.csv')
    m.meas_basis('VD')
    _, vd_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'vd_sweep_{TRIAL}.csv')
    
    
    angles = np.linspace(*SWEEP_PARAMETERS[:3])
    thetas = unp.arctan(unp.sqrt((vd_rates)/(ha_rates)))
    params = analysis.fit('line', angles, thetas)
    analysis.plot_func('line', params, angles, color='blue')
    analysis.plot_errorbar(angles, thetas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Theta Parameter (rad)')
    plt.savefig(f'{fileName}_{TRIAL}.png', dpi=600)
    # plt.show()

    x = analysis.find_value('line', params, np.pi/4, angles)
    print(f'Pi/4 at {x}')

    m.shutdown()

    # sweep around -87.5 and -42.5 smaller range for the min params :)))))
from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp

if __name__ == '__main__':
    TRIAL = 9
    #SWEEP_PARAMETERS = [65.38+3,65.38-3,20,5,3]
    SWEEP_PARAMETERS = [-100,-120,20,5,1]


    # initialize manager
    m = Manager('../config.json')

    # setup other state creation elements for state creation
    #QP_ANGLE = -20.6134
    #m.C_QP.goto(QP_ANGLE)
    #m.C_PCC.goto(4.005) # from previous calibration
    #m.B_C_HWP.goto(0)

    # If confident in current phi_plus calib, set up using config
    m.make_state('phi_plus')

    # sweep UVHWP
    m.meas_basis('HH')
    _, hh_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'hh_sweep_{TRIAL}.csv')
    m.meas_basis('HV')
    _, hv_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'hv_sweep_{TRIAL}.csv')
    m.meas_basis('VH')
    _, vh_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'vh_sweep_{TRIAL}.csv')
    m.meas_basis('VV')
    _, vv_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'vv_sweep_{TRIAL}.csv')
    m.shutdown()
    
    angles = np.linspace(*SWEEP_PARAMETERS[:3])
    thetas = unp.arctan(unp.sqrt((vh_rates + vv_rates)/(hh_rates + hv_rates)))
    params = analysis.fit('line', angles, thetas)
    analysis.plot_func('line', params, angles, color='blue')
    analysis.plot_errorbar(angles, thetas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Theta Parameter (rad)')
    plt.savefig(f'ratio_sweep_{TRIAL}.png', dpi=600)
    plt.show()

    x = analysis.find_value('line', params, np.pi/4, angles)
    print(f'Pi/4 at {x}')
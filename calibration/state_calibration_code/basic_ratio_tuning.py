from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp

if __name__ == '__main__':
    TRIAL = 4 #PLEASE UPDATE
    DATE = "05232025" #PLEASE UPDATE
    STATE = "HRVL"
    fileName = f"ratio_tuning_{DATE}_for_{STATE}" 
    
    SWEEP_PARAMETERS = [-105, -120, 15, 5, 3] #fine sweep
    #SWEEP_PARAMETERS = [-55, -85, 15, 3, 1] #coarse sweep

    # initialize manager
    m = Manager('../config.json')

    m.make_state('phi_plus')

    # parameters used for HR + e^-ipi/6 VL state
    m.log('Setting up state')
    m.B_C_HWP.goto(0)
    m.B_C_QWP.goto(-45)
    m.C_QP.goto(-19.367)

    # sweep UVHWP
    m.meas_basis('HR')
    _, hd_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'hd_sweep_{TRIAL}.csv')
    m.meas_basis('VL')
    _, va_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'va_sweep_{TRIAL}.csv')
    m.shutdown()
    
    angles = np.linspace(*SWEEP_PARAMETERS[:3])
    thetas = unp.arctan(unp.sqrt((va_rates)/(hd_rates)))
    params = analysis.fit('line', angles, thetas)
    analysis.plot_func('line', params, angles, color='blue')
    analysis.plot_errorbar(angles, thetas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Theta Parameter (rad)')
    plt.savefig(f'{fileName}_{TRIAL}.png', dpi=600)
    plt.show()

    x = analysis.find_value('line', params, np.pi/4, angles)
    print(f'Pi/4 at {x}')
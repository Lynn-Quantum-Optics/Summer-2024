from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
if __name__ == '__main__':
    TRIAL = 1
    SWEEP_PARAMETERS = [0,359,180,3,1]

    # initialize manager
    m = Manager('../config.json')

    # setup other state creation elements for state creation
    m.C_QP.goto(-20.6134)
    m.C_PCC.goto(4.005) # from previous calibration
    m.B_C_HWP.goto(0)

    # sweep UVHWP
    m.meas_basis('HH')
    _, hh_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'hh_sweep_{TRIAL}.csv')
    m.meas_basis('HV')
    _, hv_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'hv_sweep_{TRIAL}.csv')
    m.meas_basis('DD')
    _, dd_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'dd_sweep_{TRIAL}.csv')
    m.meas_basis('AD')
    _, ad_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'dd_sweep_{TRIAL}.csv')
    m.meas_basis('RR')
    _, rr_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'rr_sweep_{TRIAL}.csv')
    m.meas_basis('RL')
    _, rl_rates = m.sweep('C_UV_HWP', *SWEEP_PARAMETERS)
    m.output_data(f'rl_sweep_{TRIAL}.csv')
    m.shutdown()
    
    angles = np.linspace(*SWEEP_PARAMETERS[:3])
    h_purity = hh_rates - hv_rates
    d_purity = dd_rates - ad_rates
    r_purity = rl_rates - rr_rates
    plt.plot(angles, h_purity, label = 'HH - HV', color = 'r')
    plt.plot(angles, d_purity, label = 'DD - AD', color = 'b')
    plt.plot(angles, r_purity, label = 'RL - RR', color = 'g')
    plt.legend()
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Theta Parameter (rad)')
    plt.savefig(f'ratio_sweep_{TRIAL}.png', dpi=600)
    plt.show()
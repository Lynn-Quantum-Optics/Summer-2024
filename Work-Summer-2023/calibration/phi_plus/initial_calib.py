from lab_framework import Manager, analysis


if __name__ == '__main__':
    UVHWP_SWEEP_PARAMS = (40, 50, 20, 5, 2)


    # initialize manager
    m = Manager('../config.json')

    # do hh sweep
    m.meas_basis('HH')
    hh_angles, hh_rates = m.sweep('C_UV_HWP', *UVHWP_SWEEP_PARAMS)
    m.output_data('HH_UVHWP_sweep.py')
    
    # do vv sweep 
    m.meas_basis('VV')
    vv_angles, vv_rates = m.sweep('C_UV_HWP', *UVHWP_SWEEP_PARAMS)
    m.output_data('VV_UVHWP_sweep.py')

    # now fit the function to both
    analysis.fit('cos2_sq')
    m.shutdown()

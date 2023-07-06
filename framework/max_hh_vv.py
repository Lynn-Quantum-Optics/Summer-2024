# file to sweep QP for HH and VV settings to determine VV / HH as a function of QP angle #
if __name__ == '__main__':
        
    from core import Manager
    from os.path import join

    m = Manager()

    # define sweep params
    pos_min = 0
    pos_max = 38.3
    num_steps = 50
    num_samp = 20
    samp_period = 1 # seconds

    # do HH
    m.new_output(join('decomp_test', f'HH_{num_samp}.csv'))
    m.make_state('HH')
    m.meas_basis('HH')
    m.sweep(component='C_QP', pos_min=pos_min, pos_max=pos_max, num_steps=num_steps, num_samp=num_samp, samp_period=samp_period)

    m.close_output()

    # do VV
    m.new_output(join('decomp_test', f'VV_{num_samp}.csv'))
    m.make_state('VV')
    m.meas_basis('VV')
    m.sweep(component='C_QP', pos_min=pos_min, pos_max=pos_max, num_steps=num_steps, num_samp=num_samp, samp_period=samp_period)

    m.close_output()

    m.shutdown()
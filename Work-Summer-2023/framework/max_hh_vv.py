# file to sweep QP for HH and VV settings to determine VV / HH as a function of QP angle #
if __name__ == '__main__':
        
    from core import Manager
    from os.path import join

    # define sweep params
    pos_min = -38
    # pos_max = 38.3
    pos_max = 0
    num_steps = 50
    num_samp = 20
    samp_period = 1 # seconds

    m = Manager()
    m.make_state('HH')
    m.meas_basis('HH')

    m.new_output(join('decomp_test', f'HH_HH_{num_samp}_{pos_min}_{pos_max}.csv'))
    m.sweep(component='C_QP', pos_min=pos_min, pos_max=pos_max, num_steps=num_steps, num_samp=num_samp, samp_period=samp_period)

    m.close_output()
    m.shutdown()

    m = Manager()
    m.make_state('VV')
    m.meas_basis('VV')

    m.new_output(join('decomp_test', f'VV_VV_{num_samp}_{pos_min}_{pos_max}.csv'))
    m.sweep(component='C_QP', pos_min=pos_min, pos_max=pos_max, num_steps=num_steps, num_samp=num_samp, samp_period=samp_period)

    m.close_output()
    m.shutdown()

    # states = ['HH', 'VV']
    # bases = ['HH', 'HV', 'VH', 'VV']
    # for state in states:
    #     for basis in bases:
    #         # if (state=='HH' and basis=='HH') or (state=='VV' and basis=='VV'):
    #         print(state, basis)

    #         m = Manager()

    #         m.make_state(state)
    #         m.meas_basis(basis)

    #         m.new_output(join('decomp_test', f'{state}_{basis}_{num_samp}_{pos_min}_{pos_max}.csv'))
    #         m.sweep(component='C_QP', pos_min=pos_min, pos_max=pos_max, num_steps=num_steps, num_samp=num_samp, samp_period=samp_period)
            
    #         m.close_output()
    #         m.shutdown()
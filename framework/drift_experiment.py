from core import Manager


if __name__ == '__main__':

    # initialize manager
    m = Manager(out_file='drift_experiment_data.csv')
    m.log('initialization complete')

    # setup the state
    m.make_state('phi_plus')
    m.log(f'configured phi_plus: {m._config["state_presets"]["phi_plus"]}')

    COARSE_SAMP = (5,3)
    FINE_SAMP = (10,10)

    columns = ['HH', 'HV', 'VH', 'VV', 'HH_err', 'HV_err', 'VH_err', 'VV_err']
    coarse_df = {}
    fine_df = dict([(k,[]) for k in ])



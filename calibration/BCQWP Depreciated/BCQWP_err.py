from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    TRIAL = 6
    SWEEP_PARAMS = [-2, 2, 15, 5, 3]

    # TRIAL = 1
    # SWEEP_PARAMS = [-30, 0, 20, 5, 1]

    # TRIAL = 2
    # SWEEP_PARAMS = [-5, 10, 20, 5, 3]
    
    # TRIAL = 3
    # SWEEP_PARAMS = [-10, 10, 20, 5, 3]

    # TRIAL = 4
    # SWEEP_PARAMS = [-12, 8, 20, 5, 2]
    
    # TRIAL = 5
    # SWEEP_PARAMS = [-6, 2, 20, 5, 3]
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'BCQWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # configure the VV state
    m.make_state('VV')

    # make alice measure in V
    m.log('Sending AQWP to 0')
    m.A_QWP.goto(0)
    m.log('Sending AHWP to 45')
    m.A_HWP.goto(45)

    # put bob's wps to calibrated settings
    m.log('Sending BHWP to 0')
    m.B_HWP.goto(0)
    m.log('Sending BQWP to 0')
    m.B_QWP.goto(0)
    m.log('Sending BCHWP to 0')
    m.B_C_HWP.goto(0)

    # sweep bob's creation half waveplate
    m.log('Beginning BCQWP Sweep')

    # here we do the sweep manually
    angles = np.linspace(*SWEEP_PARAMS[:3])
    actual_angles = np.linspace(*SWEEP_PARAMS[:3]) % 360
    rates = []
    for a in actual_angles:
        m.configure_motor('B_C_QWP', a)
        rates.append(m.take_data(*SWEEP_PARAMS[3:], 'C4'))
    rates = np.array(rates)

    # save the output
    df = m.output_data(f'BCQWP_sweep{TRIAL}.csv')
    m.shutdown()
    
    '''
    df = Manager.load_data('BCQWP_sweep0.csv')
    angles, rates = df['B_C_QWP'], df['C4']
    '''

    # fitting function
    '''
    # fit the function
    params = analysis.fit('quadratic', angles, rates)

    # print fitted parameters
    print(f'Fit parameters = {params}')
    '''

    # plotting
    analysis.plot_errorbar(angles, rates, ms=0.1, fmt='ro', label='Data')
    # analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed', label=f'Fit Function')
    plt.xlabel('Bob\'s Creation QWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    # plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.savefig(f'BCQWP{TRIAL}.png', dpi=600)
    plt.show()

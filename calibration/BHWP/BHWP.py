from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TRIAL = 0
    # SWEEP_PARAMS = [-20, 20, 20, 5, 1]
    # TRIAL = 1
    # SWEEP_PARAMS = [-25, -13, 10, 5, 0.3]
    # TRIAL = 2
    # SWEEP_PARAMS = [-19, -11, 20, 5, 3]
    TRIAL = 3
    SWEEP_PARAMS = [-4, 4, 20, 5, 1]
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'BHWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # configure the VV state
    m.make_state('VV')

    # make alice measure in V
    m.log('Sending AQWP to 0')
    m.A_QWP.goto(0)
    m.log('Sending AHWP to 45')
    m.A_HWP.goto(45)
    # put bob's wps to calibrated settings
    m.log('Sending BQWP to 0')
    m.B_QWP.goto(0)
    m.log('Sending BCHWP to 0')
    m.B_C_HWP.goto(0)

    # sweep bob's creation half waveplate
    angles, rates = m.sweep('B_HWP', *SWEEP_PARAMS)

    # save the output
    df = m.output_data(f'BHWP_sweep{TRIAL}.csv')
    m.shutdown()
    '''
    df = Manager.load_data('BHWP_sweep1.csv')
    '''
    # fit the function
    params = analysis.fit('quadratic', angles, rates)

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    analysis.plot_errorbar(angles, rates, ms=0.1, fmt='ro', label='Data')
    analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed', label=f'Fit Function')
    plt.xlabel('Bob\'s HWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.savefig(f'BCHWP{TRIAL}.png', dpi=600)
    plt.show()

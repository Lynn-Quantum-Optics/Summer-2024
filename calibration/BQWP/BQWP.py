from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TRIAL = 0
    # SWEEP_PARAMS = [75, 110, 20, 5, 1]
    TRIAL = 0
    SWEEP_PARAMS = [83, 96, 20, 5, 3]
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'BQWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # configure the VV state
    m.make_state('VV')

    # make alice measure in V
    m.A_QWP.goto(0)
    m.A_HWP.goto(45)

    # sweep bob's quarter waveplate
    angles, rates = m.sweep('B_QWP', *SWEEP_PARAMS)

    # save the output
    df = m.output_data(f'BQWP_sweep{TRIAL}.csv')
    m.shutdown()
    '''
    df = Manager.load_data('BQWP_sweep1.csv')
    angles, rates = df['B_QWP'], df['C4']
    '''
    # fit the function
    params = analysis.fit('quadratic', angles, rates)

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed', label=f'Fit Function')
    analysis.plot_errorbar(angles, rates, ms=0.1, fmt='ro', label='Data')
    plt.xlabel('Bob\'s QWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    plt.title(f'Fit=${params[1].n:.3f}(x-({params[0].n:.3f}))^2 + {params[2].n:.3f}$')
    plt.savefig(f'BQWP_sweep{TRIAL}.png', dpi=600)
    plt.show()

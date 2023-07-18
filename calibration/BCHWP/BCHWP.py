from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TRIAL = 0
    # SWEEP_PARAMS = [-20, 20, 20, 5, 1]
    
    # TRIAL = 1
    TRIAL = 2
    SWEEP_PARAMS = [-4, 4, 20, 5, 3]
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'BCHWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # configure the VV state
    m.make_state('VV')

    # make alice measure in V
    m.A_QWP.goto(0)
    m.A_HWP.goto(45)
    m.B_QWP.goto(0)

    # sweep bob's creation half waveplate
    angles, rates = m.sweep('B_C_HWP', *SWEEP_PARAMS)

    # save the output
    df = m.output_data(f'BCHWP_sweep{TRIAL}.csv')
    m.shutdown()
    
    '''
    df = Manager.load_data('BCHWP_sweep1.csv')
    angles, rates = df['B_C_HWP'], df['C4']
    '''

    # fit the function
    params = analysis.fit('quadratic', angles, rates)

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    analysis.plot_errorbar(angles, rates, ms=0.1, fmt='ro', label='Data')
    analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed', label=f'Fit Function')
    plt.xlabel('Bob\'s Creation HWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.savefig(f'BCHWP{TRIAL}.png', dpi=600)
    plt.show()

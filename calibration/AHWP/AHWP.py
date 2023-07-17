from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TRIAL = 0
    # SWEEP_PARAMS = [-15, 15, 20, 5, 1]
    TRIAL = 1
    SWEEP_PARAMS = [-14, -4, 30, 5, 5]
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'AHWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # configure alice's QWP
    m.log('Sending Alice\'s QWP to calibrated zero')
    m.A_QWP.goto(0)

    # create VX
    m.log('Putting Alice\'s photon into a vertical configuration')
    m.make_state('VX')

    # sweep alice's quarter waveplate
    m.log('Sweeping AHWP')
    m.sweep('A_HWP', *SWEEP_PARAMS)

    # get the output
    df = m.output_data(f'AHWP_sweep{TRIAL}.csv')
    m.shutdown()
    '''
    df = Manager.load_data('AHWP_sweep0.csv')
    '''

    # fit the function
    params = analysis.fit('quadratic', df['A_HWP'], df['C4'])

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    analysis.plot_func('quadratic', params, df['A_HWP'], color='b', linestyle='dashed', label=f'Fit Function')
    analysis.plot_errorbar(df['A_HWP'], df['C4'], ms=0.1, fmt='r-')
    plt.xlabel('Alice\'s Measurement HWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    # plt.title('Initial A_HWP Sweep')
    plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.savefig(f'AHWP_sweep{TRIAL}.png', dpi=600)
    plt.show()

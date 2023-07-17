from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRIAL = 1
    SWEEP_PARAMS = [-8, 4, 30, 5, 5]
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'UVHWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # configure the UVHWP to produce _something_
    m.log('Sending Alice\'s QWP to calibrated zero')
    m.A_QWP.goto(0)

    # sweep alice's quarter waveplate
    m.log('Sweeping UVHWP')
    m.sweep('C_UV_HWP', *SWEEP_PARAMS)

    # get the output
    df = m.output_data(f'UVHWP_sweep{TRIAL}.csv')
    m.shutdown()
    '''
    df = Manager.load_data('UVHWP_sweep1.csv')
    '''
    # fit the function
    params = analysis.fit('quadratic', df['C_UV_HWP'], df['C4'])

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    analysis.plot_func('quadratic', params, df['C_UV_HWP'], color='b', linestyle='dashed', label=f'Fit Function')
    analysis.plot_errorbar(df['C_UV_HWP'], df['C4'], ms=0.1, fmt='r-')
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    # plt.title('Initial UVHWP Sweep')
    plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.savefig(f'UVHWP_sweep{TRIAL}.png', dpi=600)
    plt.show()

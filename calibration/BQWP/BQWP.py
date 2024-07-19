from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TRIAL = 0
    # SWEEP_PARAMS = [75, 110, 20, 5, 1]
    TRIAL = 1
    #SWEEP_PARAMS = [83, 96, 20, 5, 3]
    SWEEP_PARAMS = [-15, 15, 20, 5, 3]
    
    #'''
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
    #'''
    df = Manager.load_data(f'BQWP_sweep{TRIAL}.csv')
    angles, rates = df['B_QWP'], df['C4']
    # '''

    # fit the function
    params = analysis.fit('quadratic', angles, rates)

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # setup plot
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    
    # plotting
    analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed', label=f'${params[1].n:.3f}(x-({params[0].n:.3f}))^2 + {params[2].n:.3f}$', alpha=0.3)
    analysis.plot_errorbar(angles, rates, ms=0.1, fmt='ro', capsize=2, label='Data')

    # labels and such
    plt.title(f'Bob QWP Sweep')
    plt.xlabel('Bob\'s QWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()

    # save and show
    plt.savefig(f'BQWP_sweep{TRIAL}.png', dpi=600)
    plt.show()

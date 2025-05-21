from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TRIAL = 0
    # SWEEP_PARAMS = [-20, 20, 20, 5, 1]
    
    # TRIAL = 1
    #TRIAL = 1
    #SWEEP_PARAMS = [-12, 12, 24, 5, 3]

    #TRIAL = 2
    #SWEEP_PARAMS = [30, 50, 20, 5, 1]

    TRIAL = 3
    SWEEP_PARAMS = [0, 10, 10, 5, 3]
    
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
    
    
    df = Manager.load_data(f'BCHWP_sweep{TRIAL}.csv')
    angles, rates = df['B_C_HWP'], df['C4']
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
    plt.xlabel('Bob\'s Creation HWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    plt.title('Bob Creation HWP Sweep')
    plt.savefig(f'BCHWP_sweep{TRIAL}.png', dpi=600)
    plt.show()

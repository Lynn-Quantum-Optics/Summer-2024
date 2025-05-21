from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRIAL = 4
    SWEEP_PARAMS = [-40, -20, 20, 5, 1]
    # TRIAL = 1
    # SWEEP_PARAMS = [-25, -13, 10, 5, 0.3]
    # TRIAL = 2
    # SWEEP_PARAMS = [-19, -11, 20, 5, 3]
    # TRIAL = 0
    # SWEEP_PARAMS = [-20, 20, 20, 5, 1]
    #TRIAL = 1
    #SWEEP_PARAMS = [-25, -7, 39, 5, 3]
    #TRIAL = 2
    #SWEEP_PARAMS = [-8, 8, 20, 5, 3]
    
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
   
    df = Manager.load_data(f'BHWP_sweep{TRIAL}.csv')
    angles, rates = df['B_HWP'], df['C4']
    
    #'''
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
    plt.title('Bob HWP Sweep')
    plt.xlabel('Bob\'s HWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()

    # save and show
    plt.savefig(f'BHWP_sweep{TRIAL}.png', dpi=600)
    plt.show()

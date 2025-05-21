from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    ###### RUNNING THE SWEEP ######
    TRIAL = 11

    SWEEP_PARAMS = [93, 103, 10, 5, 3]

    # initializing manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'BCQWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # make the vv state
    m.make_state('VV')

    # make alice measure in V
    m.log('Sending AQWP to 0')
    m.A_QWP.goto(0)
    m.log('Sending AHWP to 45')
    m.A_HWP.goto(45)

    # Bob to measure in calibrated settings
    m.log('Sending BHWP to 0')
    m.B_HWP.goto(0)
    m.log('Sending BQWP to 0')
    m.B_QWP.goto(0)
    # m.log('Sending BCHWP to 0')
    # m.B_C_HWP.goto(0)

    # sweep Bob's BCQWP
    m.log("Beginning Bob's BCQWP sweep")

    # manual sweep
    angles, rates = m.sweep('B_C_QWP', *SWEEP_PARAMS)

    # save the output
    df = m.output_data(f'BCQWP_sweep{TRIAL}.csv')
    m.shutdown()

    
    ###### FITTING OUR DATA ######
    # fitting function
    params = analysis.fit('quadratic', angles, rates)

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    analysis.plot_errorbar(angles, rates, ms=0.1, fmt='ro', label='Data')
    analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed', label=f'Fit Function')

    # labels n such 
    plt.xlabel('Bob\'s Creation QWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.savefig(f'BCQWP{TRIAL}.png', dpi=600)
    plt.show()
    




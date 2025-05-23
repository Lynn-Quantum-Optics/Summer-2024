from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

# use this file to find the qp angle that minimizes the counts of a given state in the basis you expect counts to be minimized in
# will output a plot that can be used to find the "correct" qp angle

if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    SWEEP_PARAMS = [-25, -15, 10, 5, 3] 
    DATE = "05232025" #please update
    TRIAL = 6 #please update
    STATE = "HRVL"

    fileName = f"QP_sweep_{DATE}_{TRIAL}_for_{STATE}"

    # initialize the manager
    m = Manager('../config.json')

    m.make_state('phi_plus')
    m.meas_basis("VV")

    # parameters used for HR + e^-ipi/6 VL state
    m.log('Setting up state')
    m.B_C_HWP.goto(0)
    m.B_C_QWP.goto(-45)

    m.log('Setting up measurement basis')
    m.A_HWP.goto(-30)
    m.A_QWP.goto(-105)

    # check count rates
    m.log('Moving QP and UVHWP where they have been tuned to be')
    m.C_QP.goto(-19.367)
    m.C_UV_HWP.goto(-112.74443676597194)

    m.log('Checking count rates...')
    counts = m.take_data(7,5,'C4')

    # tell the user what is up
    print(f'Count rates: {counts}')

    # check if the count rates are good
    inp = input('Recalibrate QP? [y/n] ')
    if inp.lower() != 'y':
        print('Exiting...')
        m.shutdown()
        quit()

    # sweep QP
    m.log('Sweeping QP')
    angles, rates = m.sweep('C_QP', *SWEEP_PARAMS)

    # get the output
    df = m.output_data(f'{fileName}.csv')
    m.shutdown()

    df = Manager.load_data(f'{fileName}.csv')

    ###### FITTING OUR DATA ######
    # fitting function
    params = analysis.fit('quadratic', angles, rates)

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    analysis.plot_errorbar(angles, rates, ms=0.1, fmt='ro', label='Data')
    analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed', label=f'Fit Function')

    # labels n such 
    plt.xlabel('QP angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.savefig(f'{fileName}.png', dpi=600)
    plt.show()



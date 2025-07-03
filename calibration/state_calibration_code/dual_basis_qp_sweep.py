from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

# use this file to find the qp angle that produces the correct phase shift for a given state by minimizing counts in the 2 bases you expect counts to be minimized in
# will output a plot that can be used to find the "correct" qp angle

if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    SWEEP_PARAMS = [-20, -5, 15, 5, 3] 
    DATE = "07022025" #please update
    TRIAL = 1 #please update
    STATE = "ha_vd"

    fileName = f"dual_basis_QP_sweep_{DATE}_{TRIAL}_for_{STATE}"

    # initialize the manager
    m = Manager('../config.json')

    # parameters used for setting up the sstate
    m.log('Setting up state')
    m.make_state('phi_plus')
    m.B_C_HWP.goto(22.5)
    m.B_C_QWP.goto(45)
    m.C_UV_HWP.goto(-64.33913582249691)

    # m.B_C_HWP.goto(67.5)
    # m.B_C_QWP.goto(45)
    # m.C_UV_HWP.goto(-65.18826575028268)

    m.log('Setting up first measurement basis')
    m.meas_basis("AH")
    # m.meas_basis("VV")
    # m.A_HWP.goto(-7.5)
    # m.A_QWP.goto(-60)

    # sweep QP
    m.log('Sweeping QP')
    angles1, rates1 = m.sweep('C_QP', *SWEEP_PARAMS)
    df = m.output_data(f'{fileName}_basis_one.csv')

    m.log('Setting up second measurement basis')
    m.meas_basis("DV")
    # m.meas_basis("HH")
    # m.A_HWP.goto(-52.5)
    # m.A_QWP.goto(30)

    m.log('Sweeping QP')
    angles2, rates2 = m.sweep('C_QP', *SWEEP_PARAMS)
    df = m.output_data(f'{fileName}_basis_two.csv')

    # get the output
    
    m.shutdown()

    ###### FITTING OUR DATA ######
    # fitting function
    angles = angles1
    rates = rates1 + rates2

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
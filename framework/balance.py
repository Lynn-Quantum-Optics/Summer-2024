import matplotlib.pyplot as plt
from core import Manager, analysis
import pandas as pd

if __name__ == '__main__':
    
    # parameters
    GUESS = 66.39
    RANGE = 1.5
    N = 20
    SAMP = (5, 3)

    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'
    STATE = 'phi_plus'

    PCT1 = 0.50
    
    # initialize manager
    m = Manager(out_file='balance_sweep_1.csv')

    # setup the state configuration
    print(m.time, f'Configuring {STATE} state')
    m.make_state(STATE)

    # configure measurement basis
    print(m.time, f'Configuring measurement basis {BASIS1}')
    m.meas_basis(BASIS1)

    # do sweep
    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS1}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    # obtain the first round of data and switch to a new output file
    data1 = m.close_output()
    m.new_output('balance_sweep_2.csv')

    # sweep in the second basis
    print(m.time, f'Configuring measurement basis {BASIS2}')
    m.meas_basis(BASIS2)

    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS2}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    print(m.time, 'Data collected, shutting down...')
    data2 = m.close_output()
    m.shutdown()
    
    print(m.time, 'Data collection complete and manager shut down, beginning analysis...')

    args1, unc1, _ = analysis.fit('sin2', data1.C_UV_HWP, data1.C4, data1.C4_sem)
    args2, unc2, _ = analysis.fit('sin2', data2.C_UV_HWP, data2.C4, data2.C4_sem)
    x = analysis.find_ratio('sin', args1, 'sin', args2, PCT1, data1.C_UV_HWP, GUESS)

    # print result
    print(f'{COMPONENT} angle to find {PCT1*100:.2f}% coincidences ({(1-PCT1)*100:.2f}% coincidences): {x:.5f}')
    
    # plot the data and fit
    plt.title(f'Angle of {COMPONENT} to achieve {PCT1*100:.2f}% {BASIS1}\ncoincidences ({(1-PCT1)*100:.2f}% {BASIS2} coincidences): {x:.5f} degrees')
    plt.xlabel(f'{COMPONENT} angle (rad)')
    plt.ylabel(f'Count rates (#/s)')
    plt.errorbar(data1.C_UV_HWP, data1.C4, yerr=data1.C4_sem, fmt='o', label=BASIS1)
    plt.errorbar(data2.C_UV_HWP, data2.C4, yerr=data2.C4_sem, fmt='o', label=BASIS2)
    analysis.plot_func('sin2', args1, data1.C_UV_HWP, label=f'{BASIS1} fit function')
    analysis.plot_func('sin2', args2, data2.C_UV_HWP, label=f'{BASIS2} fit function')
    plt.legend()
    plt.show()

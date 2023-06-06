from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from core import Manager, analysis

if __name__ == '__main__':
    # parameters

    GUESS = 63.4
    RANGE = 3
    N = 15
    SAMP = (5, 1.5)

    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'

    PCT1 = 0.51

    
    # initialize manager
    m = Manager(out_file='balance_sweep_1.csv')

    # setup the state configuration
    print(m.time, 'configuring phi+ state')
    m.make_state('phi+')

    # configure measurement basis
    print(m.time, f'Configuring measurement basis {BASIS1}')
    m.meas_basis(BASIS1)

    # do sweep
    print(m.time, f'Beginning sweep of {BASIS1} from {GUESS-RANGE} to {GUESS+RANGE}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    # obtain the first round of data and switch to a new output file
    data1 = m.close_output()
    m.open_output('balance_sweep_2.csv')

    # sweep in the second basis
    print(m.time, f'Configuring measurement basis {BASIS2}')
    m.meas_basis(BASIS2)

    print(m.time, f'Beginning sweep of {BASIS2} from {GUESS-RANGE} to {GUESS+RANGE}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    print(m.time, 'Data collected, shutting down...')
    data2 = m.close_output()
    m.shutdown()

    print(m.time, 'Data collection complete and manager shut down, beginning analysis...')
    
    args1, unc1, _ = analysis.fit('sin', data1.C_UV_HWP, data1.C4, data1.C4_unc)
    args2, unc2, _ = analysis.fit('sin', data2.C_UV_HWP, data2.C4, data2.C4_unc)
    x = analysis.find_ratio('sin', args1, 'sin', args2, PCT1, data1.C_UV_HWP, GUESS)

    # print result
    print(f'{COMPONENT} angle to find {PCT1*100:.2f}% coincidences ({1-PCT1*100:.2f}% coincidences): {x:.5f}')
    
    # plot the data and fit
    plt.xlabel(f'{COMPONENT} angle (rad)')
    plt.ylabel(f'Count rates (#/s)')
    plt.errorbar(data1.C_UV_HWP, data1.C4, yerr=data1.C4_err, fmt='o', label=BASIS1)
    plt.errorbar(data2.C_UV_HWP, data2.C4, yerr=data2.C4_err, fmt='o', label=BASIS2)
    analysis.plot_func('sin', args1, data1.C_UV_HWP, label=f'{BASIS1} fit function')
    analysis.plot_func('sin', args2, data2.C_UV_HWP, label=f'{BASIS2} fit function')
    plt.legend()
    plt.show()

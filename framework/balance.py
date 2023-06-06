from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
# from core import Manager

if __name__ == '__main__':
    # parameters

    GUESS = np.deg2rad(63.4)
    RANGE = np.deg2rad(3)
    N = 15
    SAMP = (5, 1.5)

    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'

    PCT1 = 0.51

    
    # initialize manager
    m = Manager(out_file='balance_sweep_1.csv')

    # setup the state configuration
    print(m.time, 'Creating Phi+')
    m.configure_motors(
        C_UV_HWP = np.deg2rad(63.4),
        C_QP = np.deg2rad(-0.421),
        C_PCC = 0.0699
    )

    # configure measurement basis
    print(m.time, f'Configuring measurement basis {BASIS1}')
    m.meas_basis(BASIS1)

    # do sweep
    print(m.time, f'Beginning sweep of {BASIS1} from {GUESS-RANGE} to {GUESS+RANGE}')
    for angle in tqdm(np.linspace(GUESS-RANGE, GUESS+RANGE, N)):
        m.configure_motors(**{COMPONENT: angle})
        m.take_data(*SAMP)

    # obtain the first round of data and switch to a new output file
    data1 = m.output_data(new_output='balance_sweep_2.csv')

    # sweep in the second basis
    print(m.time, f'Configuring measurement basis {BASIS2}')
    m.meas_basis(BASIS2)

    print(m.time, f'Beginning sweep of {BASIS2} from {GUESS-RANGE} to {GUESS+RANGE}')
    for i, angle in tqdm(enumerate(np.linspace(GUESS-RANGE, GUESS+RANGE, N))):
        m.configure_motors(**{COMPONENT: angle})
        m.take_data(*SAMP)

    print(m.time, 'Data collected, shutting down...')
    data2 = m.shutdown(get_data=True)
    '''

    data1 = pd.read_csv('balance_sweep_1.csv')
    data2 = pd.read_csv('balance_sweep_2.csv')
    '''
    # put all the data together

    data = pd.DataFrame()

    data['angle 1'] = data1[f"{COMPONENT} position (rad)"]
    data['angle 2'] = data2[f"{COMPONENT} position (rad)"]

    data['counts 1'] = data1["C4 rate (#/s)"]
    data['counts 1 err'] = data1["C4 rate unc (#/s)"]

    data['counts 2'] = data2["C4 rate (#/s)"]
    data['counts 2 err'] = data2["C4 rate unc (#/s)"]

    # fit function
    def fit_func(x, a, b, c):
        return a * np.cos(x + b) + c

    # fit the data
    popt1, pcov1 = curve_fit(fit_func, data['angle 1'], data['counts 1'], sigma=data['counts 1 err'])
    popt2, pcov2 = curve_fit(fit_func, data['angle 2'], data['counts 2'], sigma=data['counts 2 err'])

    # minimize the difference between the two fits
    def min_me(x, args1, args2):
        return abs((1-PCT1)*fit_func(x, *args1) - PCT1*fit_func(x, *args2))

    sol = minimize(min_me, GUESS, args=(popt1, popt2))
    print(sol.x)

    # plot the data
    plt.xlabel(f'{COMPONENT} angle (rad)')
    plt.ylabel(f'Count rates (#/s)')
    plt.errorbar(data['angle 1'], data['counts 1'], yerr=data['counts 1 err'], fmt='o', label=BASIS1)
    plt.errorbar(data['angle 2'], data['counts 2'], yerr=data['counts 2 err'], fmt='o', label=BASIS2)
    xs = np.linspace(GUESS-RANGE, GUESS+RANGE, 100)
    plt.plot(xs, fit_func(xs, *popt1), label=f'{BASIS1} fit')
    plt.plot(xs, fit_func(xs, *popt2), label=f'{BASIS2} fit')
    plt.legend()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from core import Manager

# parameters

GUESS = np.pi/4
RANGE = 0.1
N = 50
SAMP = (5, 1.5)

COMPONENT = 'C_UV_HWP'
BASIS1 = 'HH'
BASIS2 = 'VV'

# sweep first basis
m = Manager(out_file='balance_sweep_1.csv')
m.meas_basis(BASIS1)

for i, angle in enumerate(np.linspace(GUESS-RANGE, GUESS+RANGE, N)):
    m.configure_motors(**{COMPONENT: angle})
    m.take_data(*SAMP)

data1 = m.output_data(new_output='balance_sweep_2.csv')

# sweep in the second basis

m.meas_basis(BASIS2)

for i, angle in enumerate(np.linspace(GUESS-RANGE, GUESS+RANGE, N)):
    m.configure_motors(**{COMPONENT: angle})
    m.take_data(*SAMP)

data2 = m.shutdown(get_data=True)

# put all the data together

data = pd.DataFrame()

data['angle 1'] = data1[f"{COMPONENT} position (rad)"]
data['angle 2'] = data2[f"{COMPONENT} position (rad)"]

data['counts 1'] = data1["C4 rate (#/s)"]
data['counts 1 err'] = data1["C4 rate unc (#/s)"]

data['counts 2'] = data1["C4 rate (#/s)"]
data['counts 2 err'] = data1["C4 rate unc (#/s)"]

# fit function
def fit_func(x, a, b, c):
    return a * np.cos(x + b) + c

# fit the data
popt1, pcov1 = curve_fit(fit_func, data['angle 1'], data['counts 1'], sigma=data['counts 1 err'])
popt2, pcov2 = curve_fit(fit_func, data['angle 2'], data['counts 2'], sigma=data['counts 2 err'])

# minimize the difference between the two fits
def min_me(x, args1, args2):
    return fit_func(x, *args1) - fit_func(x, *args2)

sol = minimize(fit_func, GUESS, args=(popt1, popt2))
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

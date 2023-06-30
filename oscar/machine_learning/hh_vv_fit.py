from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df_hh = pd.read_csv(join('hh_vv_data', 'UVHWP_balance_sweep1.csv'))
df_vv = pd.read_csv(join('hh_vv_data', 'UVHWP_balance_sweep_2.csv'))

plt.errorbar(df_hh['C_UV_HWP position (deg)'], df_hh['C4 rate (#/s)'], yerr = df_hh['C4 rate SEM (#/s)'], fmt='o', label='HH')
plt.errorbar(df_vv['C_UV_HWP position (deg)'], df_vv['C4 rate (#/s)'], yerr = df_vv['C4 rate SEM (#/s)'], fmt='o', label='VV')

# fitting function #
def func(x, a, b, c):
    return a* np.sin(np.deg2rad(2*x+b))**2 +c

popt_hh, pcov_hh = curve_fit(func, df_hh['C_UV_HWP position (deg)'],df_hh['C4 rate (#/s)'], sigma=df_hh['C4 rate SEM (#/s)'])

popt_vv, pcov_vv = curve_fit(func, df_vv['C_UV_HWP position (deg)'],df_vv['C4 rate (#/s)'], sigma=df_vv['C4 rate SEM (#/s)'])

angles = np.linspace(min(df_vv['C_UV_HWP position (deg)']), max(df_vv['C_UV_HWP position (deg)']), 1000)

plt.plot(angles, func(angles, *popt_hh), label='$%.3g \sin(2\\theta+%.3g)^2 + %.3g$'%(popt_hh[0], popt_hh[1], popt_hh[2]))
plt.plot(angles, func(angles, *popt_vv), label='$%.3g \sin(2 \\theta + %.3g)^2 + %.3g$'%(popt_vv[0], popt_vv[1], popt_vv[2]))

plt.title('Comparing max HH and max VV')
plt.xlabel('Angles (deg)')
plt.ylabel('C4 Coincidences')

print('HH / VV', popt_hh[0] / popt_vv[0])
s_hh = np.sqrt(np.diag(pcov_hh))[0]
s_vv = np.sqrt(np.diag(pcov_vv))[0]
print('unc',  np.sqrt((s_hh / popt_vv[0])**2 + (popt_hh[0] / (popt_vv[0])**2*s_vv)**2))

plt.legend()
plt.savefig(join('hh_vv_data', 'max_hh_vv.pdf'))
plt.show()
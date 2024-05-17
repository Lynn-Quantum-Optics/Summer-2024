from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df_vv = pd.read_csv(join('hh_vv_data', 'UVHWP_balance_sweep1.csv'))
df_hh = pd.read_csv(join('hh_vv_data', 'UVHWP_balance_sweep_2.csv'))

plt.errorbar(df_vv['C_UV_HWP position (deg)'], df_vv['C4 rate (#/s)'], yerr = df_vv['C4 rate SEM (#/s)'], fmt='o', label='VV')
plt.errorbar(df_hh['C_UV_HWP position (deg)'], df_hh['C4 rate (#/s)'], yerr = df_hh['C4 rate SEM (#/s)'], fmt='o', label='HH')

# fitting function #
def sinsq2(x, a, b, c):
    return a* np.sin(np.deg2rad(2*x+b))**2 +c
def cossq2(x, a, b, c):
    return a* np.cos(np.deg2rad(2*x+b))**2 +c

popt_vv, pcov_vv = curve_fit(sinsq2, df_vv['C_UV_HWP position (deg)'],df_vv['C4 rate (#/s)'], sigma=df_vv['C4 rate SEM (#/s)'])

popt_hh, pcov_hh = curve_fit(sinsq2, df_hh['C_UV_HWP position (deg)'],df_hh['C4 rate (#/s)'], sigma=df_hh['C4 rate SEM (#/s)'])

angles = np.linspace(min(df_hh['C_UV_HWP position (deg)']), max(df_hh['C_UV_HWP position (deg)']), 1000)

plt.plot(angles, sinsq2(angles, *popt_vv), label='$%.3g \sin(2\\theta+%.3g)^2 + %.3g$'%(popt_vv[0], popt_vv[1], popt_vv[2]))
plt.plot(angles, sinsq2(angles, *popt_hh), label='$%.3g \sin(2 \\theta + %.3g)^2 + %.3g$'%(popt_hh[0], popt_hh[1], popt_hh[2]))
# plt.plot(angles, sinsq2(angles, *popt_vv), label='$%.3g \sin(2\\theta)^2 + %.3g$'%(popt_vv[0], popt_vv[1]))
# plt.plot(angles, cossq2(angles, *popt_hh), label='$%.3g \cos(2 \\theta)^2 + %.3g$'%(popt_hh[0], popt_hh[1]))

plt.title('Comparing max VV and max HH')
plt.xlabel('Angles (deg)')
plt.ylabel('C4 Coincidences')

print('vv / hh', np.abs((popt_vv[0] + popt_vv[2]) / (popt_hh[0] + popt_hh[2])))
s_vv = np.sqrt(np.diag(pcov_vv))[0]
s_hh = np.sqrt(np.diag(pcov_hh))[0]
print('unc',  np.sqrt((s_vv / popt_hh[0])**2 + (popt_vv[0] / (popt_hh[0])**2*s_hh)**2))
print('middle point',df_vv['C4 rate (#/s)'][9] / df_hh['C4 rate (#/s)'][9])

plt.legend()
plt.savefig(join('hh_vv_data', 'max_vv_hh.pdf'))
plt.show()
import matplotlib.pyplot as plt
from lab_framework import analysis
import pandas as pd
import numpy as np
import uncertainties.unumpy as unp
import uncertainties as unc
from notify_run import Notify
# Approx. 1 min

df = pd.read_csv('phi_data_05202025.csv')
df['phi'] = df['phi'].apply(unc.ufloat_fromstr)

def poly5(x, x0, y0, a, b, c, d, e):
    return a*(x-x0)**5 + b*(x-x0)**4 + c*(x-x0)**3 + d*(x-x0)**2 + e*(x-x0) + y0


params = analysis.fit(poly5, df['QP'], df['phi'])
analysis.plot_func(poly5, params, df['QP'], color='blue', label='Poly5 Fit')
# analysis.plot_errorbar(df['QP'], df['phi'], color='red', fmt='o', ms=0.1, label='Data')
plt.errorbar(x=df['QP'], y=unp.nominal_values(df['phi']), yerr=unp.std_devs(df['phi'])*10, color='red', fmt='o', ms=0.1, label='Data (Error x 10)') #
plt.xlabel('Quartz Plate Angle (deg)')
plt.ylabel('Phi Parameter (rad) (adjusted')
plt.title('Phi Parameter vs. Quartz Plate Angle')
plt.legend()
plt.savefig('phi_sweep_05202025_forHDVA.png', dpi=600)
print(params)
plt.show()



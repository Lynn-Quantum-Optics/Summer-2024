import matplotlib.pyplot as plt
from lab_framework import analysis
import pandas as pd
import numpy as np
import uncertainties.unumpy as unp
import uncertainties as unc

# Approx. 1 min

df = pd.read_csv('purity_test_04162025_3.csv')
df['purity'] = df['purity'].apply(unc.ufloat_fromstr)

def poly5(x, x0, y0, a, b, c, d, e):
    return a*(x-x0)**5 + b*(x-x0)**4 + c*(x-x0)**3 + d*(x-x0)**2 + e*(x-x0) + y0


params = analysis.fit(poly5, df['PCC'], df['purity'])
analysis.plot_func(poly5, params, df['PCC'], color='blue', label='Poly5 Fit')
# analysis.plot_errorbar(df['QP'], df['phi'], color='red', fmt='o', ms=0.1, label='Data')
plt.errorbar(x=df['PCC'], y=unp.nominal_values(df['purity']), yerr=unp.std_devs(df['purity'])*10, color='red', fmt='o', ms=0.1, label='Data (Error x 10)')
plt.xlabel('PCC Angle (deg)')
plt.ylabel('Purity')
plt.title('Purity vs. PCC Angle')
plt.legend()
plt.savefig('purity_sweep_04162025_3.png', dpi=600)
print(params)
plt.show()
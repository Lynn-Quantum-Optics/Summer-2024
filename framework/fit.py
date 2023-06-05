import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/lynnlab/Desktop/Lynn-s-Quantum-Optics-Lab/Summer2022/AutomatedLabFramework/lab_framework/2023-06-02_14-38-03.csv")


ydata = df['C4 rate (#/s)']
yerr = df['C4 rate unc (#/s)']
xdata = df['C_QP position (rad)'].values

def func(x, a, c, d):
    return a*(np.sin(2*(x-c))**2)+d


popt, pcov = opt.curve_fit(func, xdata, ydata, sigma=yerr, absolute_sigma=True)

plt.errorbar(xdata, ydata, yerr, fmt='o')
all_x = np.linspace(xdata[0], xdata[-1], 1000)
plt.plot(all_x, func(all_x, *popt))

print(f'a = {popt[0]} +- {pcov[0][0]}')
print(f'c = {(popt[1] % (np.pi/2)) - np.pi/2} +- {pcov[1][1]}')
print(f'd = {popt[2]} +- {pcov[2][2]}')
plt.show()



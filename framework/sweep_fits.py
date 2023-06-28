from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import Manager
from measurement import meas_ab, meas_phi, meas_all
from core import analysis

def sweep_qp_phi(m:Manager, samp:Tuple[int,float], num_step:int, out_file:str=None):
    '''
    Sweep the quartz-plate to record the phi parameter as a function of angle.
    '''

    # array of angles
    angles = np.linspace(0, 45, num_step)
    
    # loop to gather phi data
    phis = []
    phi_uncs = []
    for i, x in enumerate(angles):
        m.log(f'Moving QP to {x} degrees (iteration {i+1}/{num_step})')
        m.C_QP.goto(x)
        p, pu = meas_phi(m, SAMP)
        phis.append(p)
        phi_uncs.append(pu)
    
    # repackage data into a dataframe
    df = pd.DataFrame({'QP':angles, 'phi':phis, 'unc':phi_uncs})
    df['phi'] += 360 # from experience, we want to start with phi > 0

    # this is a silly bit of code removes any discontinuities
    for i in range(1, len(df)):
        if abs(df['phi'][i] - df['phi'][i-1]) > 180:
            df['phi'][i:] -= 360

    # save the data if an output file was provided
    if out_file is not None:
        df.to_csv(out_file, index=False)

    # error bar plot of the data
    plt.errorbar(df['QP'], df['phi'], df['unc'], fmt='ro', ms=0.1)

    # fit a secant function to the datas!
    params, uncs = analysis.fit('sec', df['QP'], df['phi'], df['unc'], p0=[-1000, 500], bounds=([-1e5, -2000], [0, 2000]))
    analysis.plot_func('sec', params, df['QP'], label=f'${params[0]:.3f}\\sec(\\theta) + {params[1]:.3f}$')
    
    # plot axes and such
    plt.xlabel('Quartz Plate Angle (degrees)')
    plt.ylabel('Phi Parameter (degrees)')
    plt.title(f'Phi parameter as quartz plate\nis swept from {df["QP"].min()} to {df["QP"].max()} degrees')
    plt.legend()

    # print the results and show the plot
    print(f'a = {params[0]} ± {uncs[0]}')
    print(f'b = {params[1]} ± {uncs[1]}')
    plt.show()




if __name__ == '__main__':
    
    SAMP = (3,0.5)
    NUM_STEP = 30

    sweep_qp_phi(None, SAMP, NUM_STEP, None)

    
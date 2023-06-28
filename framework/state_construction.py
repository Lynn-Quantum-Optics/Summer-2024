
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import Manager
from state_measurement import meas_ab, meas_phi, meas_all

if __name__ == '__main__':
    
    SAMP = (3,0.5)
    NUM_STEP = 30
    
    m = Manager()

    angles = np.linspace(0, 40, NUM_STEP)
    phis = []
    phi_uncs = []
    for i, x in enumerate(angles):
        m.log(f'Moving QP to {x} degrees (iteration {i+1}/{NUM_STEP})')
        m.C_QP.goto(x)
        p, pu = meas_phi(m, SAMP)
        phis.append(p)
        phi_uncs.append(pu)
    
    df = pd.DataFrame({'QP':angles, 'phi':phis, 'unc':phi_uncs})
    df.to_csv('qp_phi_sweep.csv', index=False)
    '''
    df = pd.read_csv('qp_phi_sweep.csv')
    df['phi'] = df['phi'] % 180
    '''
    plt.errorbar(df['QP'], df['phi'], df['unc'], fmt='ro')
    plt.xlabel('Quartz Plate Angle (degrees)')
    plt.ylabel('Phi Parameter (degrees)')
    plt.title(f'Phi parameter as quartz plate is swept\nfrom {angles.min()} to {angles.max()}')
    plt.show()

# file to test the results of my jones decomp method

import numpy as np
import pandas as pd
from core import Manager
from full_tomo import get_rho

# read in angle settings
df = pd.read_csv('../oscar/machine_learning/decomp/ertias.csv')

# set up manager #
SAMP = (5, 1) # sampling parameters
m = Manager()
m.new_output('decomp_data.csv')

# define states of interest
states = [('PhiP')]
for state in states:
    # get angles
    UV_HWP_theta = df.loc[(df['state'] == state) & (df['setup']=='C0')]['UV_HWP'].values[0]
    QP_phi = df.loc[(df['state'] == state) & (df['setup']=='C0')]['QP'].values[0]
    B_HWP_theta = df.loc[(df['state'] == state) & (df['setup']=='C0')]['B_HWP'].values[0]

    print(UV_HWP_theta, QP_phi, B_HWP_theta)

    # set angles
    m.configure_motors(
        C_UV_HWP=UV_HWP_theta,
        C_QP = QP_phi,
        B_C_HWP = B_HWP_theta
    )

    # get the density matrix
    rho, unc = get_rho(m, SAMP)

    # print results
    print('RHO\n---')
    print(rho)
    print('UNC\n---')
    print(unc)

    # save results
    with open('rho_out.npy', 'wb') as f:
        np.save(f, (rho, unc))

# close out
m.close_output()
m.shutdown()


if __name__ == '__main__':

    # file to test the results of my jones decomp method

    import numpy as np
    import pandas as pd
    import os

    from lab_framework import Manager
    from full_tomo_updated_richard import *

    import sys
    sys.path.insert(0, '../oscar/machine_learning')
    from rho_methods import get_fidelity, get_purity
    from sample_rho import PhiP, PhiM, PsiM, PsiP, get_E0

    # set up manager #
    SAMP = (5, 1) # (num measurements per basis, num seconds per meas)
    m = Manager()
    tnum = 35 # trial number

    expt = int(input('0 to run Phip and PsiP, 1 for E0(60, 36), 2 for E0 eta = 45, 60: '))

    # define states of interest #


    # Bell
    if expt==0:
        df = pd.read_csv('../oscar/machine_learning/decomp/bell_0.999_neg.csv')
        # states_names = [('PhiP',), ('PsiP',)]
        # states = [PhiP, PsiP]
        states_names = [('PsiM',)]
        states = [PsiM]

    elif expt==1 or expt==2:
        df = pd.read_csv('../oscar/machine_learning/decomp/e0_neg_ci.csv')
        if expt==1:
            eta_ls = [np.pi/3]
            chi_ls = [np.pi/5]

        else:
            # E0 states
            # fit eta at variety of values
            eta_ls =[np.pi/6]
            chi_ls = np.linspace(0, np.pi/2, 6)
        states_names = []
        states = []
        for eta in eta_ls:
            for chi in chi_ls:
                states_names.append(('E0', (np.rad2deg(eta), np.rad2deg(chi))))
                states.append(get_E0(eta, chi))

    for i, state_n in enumerate(states_names):
        state = states[i]
        print(state_n, len(state_n))

        if len(state_n) == 1:
            UV_HWP_theta = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0')]['UV_HWP'].values[0])
            QP_phi = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0')]['QP'].values[0])
            B_HWP_theta = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0')]['B_HWP'].values[0])

        elif len(state_n) == 2: # has angle settings
            eta = np.round(state_n[1][0],5)
            chi = np.round(state_n[1][1],5)

            print(eta, chi)

            UV_HWP_theta = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0') & (df['eta']==eta) & (df['chi']==chi)]['UV_HWP'].values[0])
            QP_phi = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0') & (df['eta']==eta) & (df['chi']==chi)]['QP'].values[0])
            B_HWP_theta = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0') & (df['eta']==eta) & (df['chi']==chi)]['B_HWP'].values[0])

        print('UV_HWP', UV_HWP_theta)
        print('QP', QP_phi)
        print('B_C_HWP', B_HWP_theta)

        # set angles
        m.configure_motors(
            C_UV_HWP=UV_HWP_theta,
            C_QP = QP_phi,
            B_C_HWP = B_HWP_theta,
            C_PCC = 4.005 # optimal value from phi_plus in config
        )

        # get the density matrix
        rho, unc, Su, un_proj, un_proj_unc = get_rho(m, SAMP)

        # print results
        print('measured rho\n---')
        print(rho)
        print('actual rho\n ---')
        print(state)

        # compute fidelity
        fidelity = get_fidelity(rho, state)
        print('fidelity', fidelity)
        purity = get_purity(rho)
        print('purity', purity)

        angles = [UV_HWP_theta, QP_phi, B_HWP_theta]

        # save results
        with open(f'decomp_test/rho_{state_n}_{tnum}.npy', 'wb') as f:
            np.save(f, (rho, unc, Su, un_proj, un_proj_unc, state, angles, fidelity, purity))

    # close out
    df = m.output_data(f'decomp_test/decomp_data_{tnum}.csv')
    m.shutdown()
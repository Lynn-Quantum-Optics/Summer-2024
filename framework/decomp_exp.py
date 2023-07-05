if __name__ == '__main__':

    # file to test the results of my jones decomp method

    import numpy as np
    import pandas as pd
    import os

    from core import Manager
    from full_tomo import get_rho

    import sys
    sys.path.insert(0, '../oscar/machine_learning')
    from rho_methods import get_fidelity, get_purity
    from sample_rho import PhiP, PhiM, PsiM, PsiP, get_E0

    # read in angle settings
    # df = pd.read_csv('../oscar/machine_learning/decomp/bell_0.999.csv')
    df = pd.read_csv('../oscar/machine_learning/decomp/ertias_2_f.csv')

    # set up manager #
    SAMP = (5, 1) # (num measurements per basis, num seconds per meas)
    m = Manager()
    tnum = 23 # trial number
    m.new_output(f'decomp_test/decomp_data_{tnum}.csv')

    # define states of interest #

    # Bell
    # states_names = [('PhiP',), ('PsiP',)]
    # states = [PhiP, PsiP]

    # E0 states
    # fit eta at 45 degrees
    # eta = np.pi / 4
    eta = np.pi/3
    # chi_ls = np.linspace(0, np.pi/2, 6)
    chi_ls = [-np.pi/2]
    states_names = [('E0', (np.rad2deg(eta), np.rad2deg(chi))) for chi in chi_ls]
    states = [get_E0(eta, chi) for chi in chi_ls]

    for i, state_n in enumerate(states_names):
        state = states[i]

        if len(state_n) == 1:
            UV_HWP_theta = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0')]['UV_HWP'].values[0])
            QP_phi = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0')]['QP'].values[0])
            B_HWP_theta = float(df.loc[(df['state'] == state_n[0]) & (df['setup']=='C0')]['B_HWP'].values[0])

        elif len(state_n) == 2: # has angle settings
            eta = state_n[1][0]
            chi = state_n[1][1]

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

        # m.make_state('phi_plus')

        # get the density matrix
        rho, unc, Su = get_rho(m, SAMP)

        # print results
        print('measured rho\n---')
        print(rho)
        print('uncertainty \n---')
        print(unc)
        print('actual rho\n ---')
        print(state)

        # compute fidelity
        fidelity = get_fidelity(rho, state)
        print('fidelity', fidelity)
        purity = get_purity(rho)
        print('purity', purity)

        # save results
        with open(f'decomp_test/rho_{state_n}_{tnum}.npy', 'wb') as f:
            np.save(f, (rho, unc, Su, state, fidelity, purity))

    # close out
    m.close_output()
    m.shutdown()


    # # params for GD #
    # N = 5 # number of times to do it
    # lr = 0.7 # learning rate
    # epsilon = .95 # exit if we have >= this fidelity

    # for i, state_n in enumerate(states_names):
    #     global RHO, UNC, ANGLES
    #     # get actual targ state
    #     state = states[i]

    #     def loss_fidelity(angles):
    #         ''' Function to minimize.
    #         Params:
    #             angles: for each component, initial guess by jones decomp
    #             returns: 1- sqrt(fidelity)
    #         '''
    #         UV_HWP_theta, QP_phi, B_HWP_theta = angles

    #         # set angles
    #         m.configure_motors(
    #             C_UV_HWP=UV_HWP_theta,
    #             C_QP = QP_phi,
    #             B_C_HWP = B_HWP_theta
    #         )

    #         m.C_UV_HWP.goto(UV_HWP_theta)
    #         m.C_QP.goto(QP_phi)
    #         m.B_C_HWP.goto(B_HWP_theta)

    #         # get the density matrix
    #         rho, unc = get_rho(m, SAMP)

    #         # print results
    #         print('RHO\n---')
    #         print(rho)
    #         print('UNC\n---')
    #         print(unc)

    #         # set globals
    #         RHO = rho
    #         UNC = unc
    #         ANGLES = angles

    #         # compute fidelity
    #         fidelity = get_fidelity(rho, state)
    #         print('fidelity', fidelity)
    #         return 1-np.sqrt(fidelity)

    #     # def minimize_loss(x0):
    #     #     '''Function that performs minimization based on input'''
    #     #     # set bounds (in deg) for components
    #     #     bounds = [(0, 45), (-38.57, 38.57), (0, 45)]
    #     #     result = minimize(loss_fidelity, x0=x0, bounds=bounds)
    #     #     return result.x # return the optimal components


    #     # get angles
    #     angles = [df.loc[(df['state'] == state_n) & (df['setup']=='C0')]['UV_HWP'].values[0], df.loc[(df['state'] == state_n) & (df['setup']=='C0')]['QP'].values[0],df.loc[(df['state'] == state_n) & (df['setup']=='C0')]['B_HWP'].values[0]]

    #     # run initial result
    #     loss = loss_fidelity(angles)
    #     # gradient
    #     grad = approx_fprime

    #     # do GD optimization
    #     for i in range(N):



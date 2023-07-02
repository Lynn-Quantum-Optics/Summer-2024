if __name__ == '__main__':

    # file to test the results of my jones decomp method

    import numpy as np
    import pandas as pd
    from scipy.optimize import approx_fprime, minimize

    from core import Manager
    from full_tomo import get_rho

    import sys
    sys.path.insert(0, '../oscar/machine_learning')
    from rho_methods import get_fidelity
    from sample_rho import PhiP

    # read in angle settings
    df = pd.read_csv('../oscar/machine_learning/decomp/bell_0.999.csv')

    # set up manager #
    SAMP = (5, 1) # (num measurements per basis, num seconds per meas)
    m = Manager()
    m.new_output('decomp_test/decomp_data_t3.csv')

    # define states of interest
    states_names = [('PhiP')]
    states = [PhiP]

    for i, state_n in enumerate(states_names):
        state = states[i]

        UV_HWP_theta = float(df.loc[(df['state'] == state_n) & (df['setup']=='C0')]['UV_HWP'].values[0])
        QP_phi = float(df.loc[(df['state'] == state_n) & (df['setup']=='C0')]['QP'].values[0])
        PCC_theta = float(df.loc[(df['state'] == state_n) & (df['setup']=='C0')]['PCC'].values[0])
        B_HWP_theta = float(df.loc[(df['state'] == state_n) & (df['setup']=='C0')]['B_HWP'].values[0])

        # set angles
        m.configure_motors(
            C_UV_HWP=UV_HWP_theta,
            C_QP = QP_phi,
            B_C_HWP = B_HWP_theta,
            C_PCC = PCC_theta
        )

        # get the density matrix
        rho, unc = get_rho(m, SAMP)

        # print results
        print('RHO\n---')
        print(rho)
        print('UNC\n---')
        print(unc)

        # compute fidelity
        fidelity = get_fidelity(rho, state)
        print('fidelity', fidelity)

        # save results
        with open(f'decomp_test/rho_{state_n}.npy', 'wb') as f:
            np.save(f, (rho, unc))

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



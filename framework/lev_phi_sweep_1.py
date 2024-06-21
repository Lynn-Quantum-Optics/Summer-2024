from lab_framework import Manager
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
from full_tomo_updated_richard import get_rho
from analysis_old import *
import pandas as pd
import scipy.linalg as la

sys.path.insert(0, '../oscar/machine_learning')
from rho_methods import get_fidelity, get_purity

## Oscar rho_methods file in Summer 2023 ##
def get_purity(rho):
    ''' Calculates the purity of a density matrix. '''
    return np.real(np.trace(rho @ rho))

def get_fidelity(rho1, rho2):
    '''Compute fidelity of 2 density matrices'''
    return np.real((np.trace(la.sqrtm(la.sqrtm(rho1)@rho2@la.sqrtm(rho1))))**2)
##                                     ##

"""
Procedure:
1. run this program to set creation state to phi plus and setup measurement polarizations for alice and bob
2. sweep/use other optimization method to turn quartz plate to minimize counts <- **use a sweep for both QP and HWP**
3. run program to alternate between HH and VV measurements then turn the UVHWP to give the correct rate ratio by sweeping **utilize balance.py for the UVHWP**
4. turn BCHWP to flip H and V <- in the very specific case where we make Psi plus, move BCHWP first, this issue will only be caused by Psi plus

use ratio of sin and cos to create initial guess for UVHWP and QP range to guess phi
check first on alpha = 0 so that the state is psi plus
Check with full tomography at the very end (look into previous scripts)

Specifically ONLY for psi plus
1. make HH + e^(i*pi) VV with BCHWP at 0 degrees then set BCHWP to 45 degrees after setting QP and UVHWP

any angle between 45 to 90 degrees is ok, in first octant between 0 and 45 degrees signs flip
the octant alternate between flipping and not flipping

angles for full tomography:
    QP: -33.5 degrees
    UVHWP: 56.48811 degrees

Results:
    theoretical rho:
    [0  0   0   0]
    [0  0.71650635  0.25 + 0.375i   0]
    [0  0.25 - 0.375i   0.28349365  0]
    [0  0   0   0]

    experimental rho:
    [0.00394011 - 0.00768909 + 0.00967013i   -0.00622985 + 0.01479892i   -0.00957832 + 0.0015256i]
    [-0.00768909 - 0.00967013i  0.71380471  0.2080949 - 0.37910358i -0.01783937 - 0.0212087i]
    [-0.00622985 - 0.01479892i  0.208949 + 0.37910358i  0.27767032  -0.0075087 - 0.01341226i]
    [-0.00957832 - 0.0015256i   -0.01783937 + 0.0212087i    -0.0075087 + 0.01341226i    0.00458486]

    new exp rho:
    RHO
    ---
    [[ 0.00425164+0.j         -0.00674868+0.00695096j  0.0029137 -0.00920168j
   0.0031735 +0.00981046j]
 [-0.00674868-0.00695096j  0.70670894+0.j          0.30183819+0.31566753j
  -0.01175356-0.01475974j]
 [ 0.0029137 +0.00920168j  0.30183819-0.31566753j  0.2849319 +0.j
  -0.00347926-0.00509636j]
 [ 0.0031735 -0.00981046j -0.01175356+0.01475974j -0.00347926+0.00509636j
   0.00410752+0.j        ]]

   it worked

    uncertainty for diagonals:
    0.00470298 for all entries

points to test: oscar fixed alpha at pi/4 and pi/3, swept beta

"""

"""
If the minimum is 0, we need to compare it to other plot to see the counts to see if the counts are the same as another minimum since 0 is always
a local minimum. If 0 isn't a minimum, we don't need to worry about this.

this means that for psi_plus, likely will have to manually set the QP to 0 before running the UVHWP sweep

Comparing full tomography density matrix and theoretical density matrix:
diagonal elements should be very close to one another, off diagonals will be close but with a smaller magnitude (around 95% of the value), if diagonals are 0
they might be nonzero
"""

def get_params(alpha, beta):
    """
    Take in two angles alpha and beta in radians where the created state is cos(alpha)*Phi_plus + (e^i*beta)*sin(alpha)*Phi_minus
    and returns the measurement angles that the HWP and QWP need to be set at per the notes for state creation.

    Parameters
    ---------
    alpha, beta: float
        Angles in radians between 0 and 2*pi

    Returns
    -------
    meas_HWP_angle: float
        The angle that Bob's measurement HWP should be turned to
    meas_QWP_angle: float
        The angle that Bob's measurement QWP should be turned to

    """

    # calculates phi depending on different values of alpha and beta, where the desired creation state is cos(theta)*HV + (e^i*phi)*sin(theta)*VH

    if alpha <= math.pi/2 and beta <= math.pi/2:
        r1 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        delta = math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1))
        gamma = math.asin(-(math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2)) + 2*math.pi
        phi = gamma - delta

    if alpha >= math.pi/2 and beta >= math.pi/2:
        r1 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        delta = math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1))
        gamma = math.pi + math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2))
        phi = gamma - delta

    if alpha <= math.pi/2 and beta >= math.pi/2:
        r1 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        delta = math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1))
        gamma = math.asin(-(math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2)) + 2*math.pi
        phi = gamma - delta
        
    if alpha >= math.pi/2 and beta <= math.pi/2:
        r1 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        delta = math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1))
        gamma = math.pi + math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2))
        phi = gamma - delta

    # calculate theta based on alpha and beta
    HH_frac = (1+math.cos(beta)*math.sin(2*alpha))/2
    theta = math.acos(math.sqrt((1+math.cos(beta)*math.sin(2*alpha))/2))

    # find angles b and u, which determine the angle Bob's measurement waveplates should be oriented
    # introduces a phase shift of pi to the angle phi because of Bob's creation HWP introducing a negative sign, which means the experimentally
    # desired creation state is cos(theta)*HV - (e^i*phi)*sin(theta)*VH
    # phi = phi + np.pi ; note, use for creating psi.
    b = np.pi/4
    u = (phi + np.pi)/2

    rate_ratio = (math.tan(theta))**2

    # find the measurement wave plate angles in terms of b and u
    meas_HWP_angle = (b + u) / 2
    meas_QWP_angle = u + math.pi/2

    return [meas_HWP_angle,meas_QWP_angle,HH_frac]

def QP_sweep(m:Manager, HWP_angle, QWP_angle, num):
    '''
    Performs a QP sweep to determine the angle that the QP needs to be set at for state creation. Finds the angle that minimizes counts.

    Parameters
    ----------
    m: Manager class object
    HWP_angle: float
        The angle at which Bob's measurement HWP should be set in radians
    QWP_angle: float
        The angle at which Bob's measurement QWP should be set in radians

    '''

    # set the output file for manager
    # m.new_output(f"int_state_sweep_WP3_2_phi45_sweep/QP_sweep_{num}.csv")
    # find a way to name file with alpha and beta

    # set the creation state to phi plus
    print(m.time, "Setting creation state to phi plus")
    m.make_state('phi_minus')
    m.log(f'configured phi_minus: {m._config["state_presets"]["phi_minus"]}')

    # turn alice's measurement plates to measure (H+V)/sqrt(2)
    print(m.time, "Turning Alice's measurement plates")
    m.configure_motors(A_HWP = np.rad2deg(np.pi/8), A_QWP = np.rad2deg(np.pi/2))

    # turn bob's measurement plates to measure H/sqrt(2) - (e^i*phi)*V/sqrt(2)
    print(m.time, "Turning Bob's measurement plates")
    m.configure_motors(B_HWP = np.rad2deg(HWP_angle), B_QWP = np.rad2deg(QWP_angle))

    # sweep the QP to determine the minimum count angle
    # sweeps through negative angles so that laser reflection points inward, if the counts are higher when the QP sweeps the other way, sweep positive
    m.sweep("C_QP", -35, -20, 20, 5, 3) #Sometimes the minimum is near the edge of the bounds in which case you won't get a parabola/normal angle. 

    print(m.time, "Sweep complete")

    # read the data into a dataframe
    df = m.output_data(f"int_state_sweep_WP3_2/QP_sweep_{num}.csv")
    data = pd.read_csv(f"int_state_sweep_WP3_2/QP_sweep_{num}.csv")
    
    # shuts down the manager
    # m.shutdown()

    # take the counts of the quartz sweep at each angle and find the index of the minimum data point
    QP_counts = data["C4"]
    print('QP Counts', QP_counts)
    min_ind = 0
    for i in range(len(QP_counts)):
        if QP_counts[i] == min(QP_counts):
            min_ind = i
    
    # creates a new data set of counts centered around the previous sweep's minimum data point
    new_guess = data["C_QP"][min_ind]
    RANGE = 5

    # finds the new minimum and maximum indices of the truncated data set
    min_bound = 0
    max_bound = len(QP_counts)

    # sets the minimum and maximum indices to the data point with the angle value (in degrees) new_guess +- RANGE
    for i in range(len(QP_counts)):
        if data["C_QP"][i] <= new_guess - RANGE:
            min_bound = i
        if data["C_QP"][i] >= new_guess + RANGE:
            max_bound = i
            break
    
    # create new truncated data set using min_bound and max_bound
    fit_data = QP_counts[min_bound:max_bound]
    fit_angles = data["C_QP"][min_bound:max_bound]
    fit_unc = data["C4_SEM"][min_bound:max_bound]
    print('fit data:', fit_data)
    print('fit angles:', fit_angles)
    print('fit unc:', fit_unc)
    # fits the truncated data set to a quartic fit function
    args1, unc1 = fit('quartic', fit_angles, fit_data, fit_unc)

    # plt.title('Angle of QP to minimize counts')
    # plt.xlabel('QP angle (deg))')
    # plt.ylabel('Count rates (#/s)')
    # plt.errorbar(fit_angles, fit_data, yerr=fit_unc, fmt='o', label="Measurement Basis")
    # analysis.plot_func('quartic', args1, fit_angles, label='Measurement fit function')
    # plt.legend()
    # plt.show()

    # finds the angle that corresponds to the minimum value of the fit function
    def fit_func(x):

        return args1[0] * x**4 + args1[1] * x**3 + args1[2] * x**2 + args1[3] * x + args1[4]

    # finds the angle at which the minimum of the fit function occurs to return as the QP angle setting
    minimum = opt.minimize(fit_func, new_guess)
    min_angle = minimum.pop('x')

    # prints and returns the angle
    print(min_angle)
    return min_angle[0]

def UVHWP_sweep(m:Manager, ratio, num):
    """
    Adapted from Alec's code in balance.py
    """

    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'

    # sweep in second octant because the standard phi plus state setting for the UV_HWP is there
    GUESS = -111.398
    RANGE = 22.5 # Something that could mess up.
    N = 20
    SAMP = (5, 3)

    PCT1 = ratio

    # m.new_output(f"int_state_sweep_WP3_2/UVHWP_balance_sweep1_{num}.csv")

    # configure measurement basis
    print(m.time, f'Configuring measurement basis {BASIS1}')
    m.meas_basis(BASIS1)

    # do sweep
    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS1}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    # obtain the first round of data and switch to a new output file
    df1 = m.output_data(f"int_state_sweep_WP3_2/UVHWP_balance_sweep1_{num}.csv")
    data1 = pd.read_csv(f"int_state_sweep_WP3_2/UVHWP_balance_sweep1_{num}.csv")
    # data1 = m.close_output()
    # m.new_output(f'int_state_sweep_WP3_2/UVHWP_balance_sweep2_{num}.csv')

    # sweep in the second basis
    print(m.time, f'Configuring measurement basis {BASIS2}')
    m.meas_basis(BASIS2)

    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS2}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    print(m.time, 'Data collected, shutting down...')
    # data2 = m.close_output()
    df2 = m.output_data(f'int_state_sweep_WP3_2/UVHWP_balance_sweep2_{num}.csv')
    data2 = pd.read_csv(f'int_state_sweep_WP3_2/UVHWP_balance_sweep2_{num}.csv')
    
    print(m.time, 'Data collection complete and manager shut down, beginning analysis...')
    # m.shutdown()

    args1, unc1 = fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_SEM)
    args2, unc2 = fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_SEM)
    x = find_ratio('sin2_sq', args1, 'sin2_sq', args2, PCT1, data1.C_UV_HWP, GUESS)

    # print result
    print(f'{COMPONENT} angle to find {PCT1*100:.2f}% coincidences ({(1-PCT1)*100:.2f}% coincidences): {x:.5f}')
    
    # plot the data and fit
    # plt.title(f'Angle of {COMPONENT} to achieve {PCT1*100:.2f}% {BASIS1}\ncoincidences ({(1-PCT1)*100:.2f}% {BASIS2} coincidences): {x:.5f} degrees')
    # plt.xlabel(f'{COMPONENT} angle (deg)')
    # plt.ylabel(f'Count rates (#/s)')
    # plt.errorbar(data1.C_UV_HWP, data1.C4, yerr=data1.C4_sem, fmt='o', label=BASIS1)
    # plt.errorbar(data2.C_UV_HWP, data2.C4, yerr=data2.C4_sem, fmt='o', label=BASIS2)
    # analysis.plot_func('sin2_sq', args1, data1.C_UV_HWP, label=f'{BASIS1} fit function')
    # analysis.plot_func('sin2_sq', args2, data2.C_UV_HWP, label=f'{BASIS2} fit function')
    # plt.legend()
    # plt.show()

    return x

# methods for finding the density matrix of a given state
def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

# Gets the density matrix of the state given some alpha and beta from above.
def get_theo_rho(alpha, beta):

    H = ket([1,0])
    V = ket([0,1])


    PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)
    phi = np.cos(alpha)*PHI_PLUS + np.exp(1j*beta)*np.sin(alpha)*PHI_MINUS

    rho = phi @ phi.conj().T

    #PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    #PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)
    #psi = np.cos(alpha)*PSI_PLUS + np.exp(1j*beta)*np.sin(alpha)*PSI_MINUS
    #rho = psi @ psi.conj().T

    return rho

def state_tomo(m, C_UV_HWP_ang, C_QP_ang, B_C_HWP_ang):
    '''
    Runs a full tomography
    '''
    SAMP = (5, 1)

    # set new output file
    m.new_output('int_state_tomography_data.csv')

    # load configured state
    m.C_UV_HWP.goto(C_UV_HWP_ang)
    m.C_QP.goto(C_QP_ang)
    m.B_C_HWP.goto(B_C_HWP_ang)

    # get the density matrix
    rho, unc, _ = get_rho(m, SAMP)

    # print results
    print('RHO\n---')
    print(rho)
    print('UNC\n---')
    print(unc)

    # save results
    with open('rho_out.npy', 'wb') as f:
        np.save(f, (rho, unc))

    m.close_output()

if __name__ == '__main__':
    # makes psi+ state, sweeping over phase values
    alphas = [np.pi/4]
    betas = np.linspace(0.001, np.pi/2, 6)
    states_names = []
    states = []

    for alpha in alphas:
        for beta in betas:
            states_names.append((np.rad2deg(alpha), np.rad2deg(beta)))
            states.append((alpha, beta))

        # select only data points3:]
    #states = states[5:]
    #states_names= states_names[5:]
    
    SAMP = (5, 1)
    m = Manager()

    # main loop for iterating over states
    for i, state_n in enumerate(states_names):
        state = states[i]

        """
        Note these angles are all in radians, not degrees.
        """

        meas_HWP_angle, meas_QWP_angle, HH_frac = get_params(state[0], state[1])

        C_QP_angle = QP_sweep(m,meas_HWP_angle,meas_QWP_angle, i) # +4 added when apparatus bugged after 4 trials taken

        m.C_QP.goto(C_QP_angle)

        UVHWP_angle = UVHWP_sweep(m, HH_frac, i)

        # m.new_output(f'int_state_sweep_WP3_2/sweep_data_{state}.csv')

        B_C_HWP_angle = 67.5 #0, 45 -> 0 to change from psi to phi 

        m.configure_motors(
            C_UV_HWP=UVHWP_angle,
            C_QP = C_QP_angle,
            B_C_HWP = B_C_HWP_angle,
            C_PCC =  3.7894 # optimal value from phi_plus in config
        )

        # get the density matrix
        rho, unc, Su, un_proj, un_proj_unc = get_rho(m, SAMP)

        # theoretical density matrix
        actual_rho = get_theo_rho(state[0], state[1])

        # print results
        print('measured rho\n---')
        print(rho)
        print('uncertainty \n---')
        print(unc)
        print('actual rho\n ---')
        print(actual_rho)

        # compute fidelity
        fidelity = get_fidelity(rho, actual_rho)
        print('fidelity', fidelity)
        purity = get_purity(rho)
        print('purity', purity)

        angles = [UVHWP_angle, C_QP_angle, 67.5] # the 0 is B_C_HWP angle

        # save results
        with open(f"int_state_sweep_WP3_2/rho_('E0', {state_n})_1.npy", 'wb') as f:
            np.save(f, (rho, unc, Su, un_proj, un_proj_unc, state, angles, fidelity, purity))
        date="6032024"
        tomo_df = m.output_data(f'int_state_sweep_WP3_2/tomo_data_{state}_{date}.csv')
        # m.close_output()
        
    m.shutdown()


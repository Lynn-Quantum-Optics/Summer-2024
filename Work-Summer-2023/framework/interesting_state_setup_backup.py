from core import Manager, analysis
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys

sys.path.insert(0, '../oscar/machine_learning')
from rho_methods import get_fidelity, get_purity

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
    QP: -20 degrees
    UVHWP: 60.16582 degrees

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
    [[ 0.00468262+0.j         -0.02298898+0.00989265j  0.00953237+0.01825194j
    -0.00841357-0.00810226j]
    [-0.02298898-0.00989265j  0.70648134+0.j         -0.29334017-0.3166438j
    -0.02279506-0.02822054j]
    [ 0.00953237-0.01825194j -0.29334017+0.3166438j   0.28363312+0.j
    0.00928785-0.00978311j]
    [-0.00841357+0.00810226j -0.02279506+0.02822054j  0.00928785+0.00978311j
    0.00520291+0.j        ]]

    uncertainty for diagonals:
    0.00470298 for all entries

points to test:

"""

def get_params(alpha, beta):
    """
    Take in two angles alpha and beta in radians where the created state is cos(alpha)*Psi_plus + (e^i*beta)*sin(alpha)*Psi_minus
    and returns the measurement angles that the HWP and QWP need to be set at per the notes for state creation.

    
    """

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
    # phi = phi + np.pi
    b = np.pi/4
    u = (phi + np.pi)/2

    rate_ratio = (math.tan(theta))**2

    meas_HWP_angle = (b + u) / 2
    meas_QWP_angle = u + math.pi/2

    return [meas_HWP_angle,meas_QWP_angle,HH_frac]

def QP_sweep(m:Manager, HWP_angle, QWP_angle):
    '''
    Performs a QP sweep to determine the angle that the QP needs to be set at for state creation. Finds the angle that minimizes counts.
    '''

    # set the output file for manager
    m.new_output(f"QP_sweep_{HWP_angle}_{QWP_angle}.csv")
    # find a way to name file with alpha and beta

    # set the creation state to phi plus
    print(m.time, "Setting creation state to phi plus")
    m.make_state('phi_plus')
    m.log(f'configured phi_plus: {m._config["state_presets"]["phi_plus"]}')

    # turn alice's measurement plates to measure (H+V)/sqrt(2)
    print(m.time, "Turning Alice's measurement plates")
    m.configure_motors(A_HWP = np.rad2deg(np.pi/8), A_QWP = np.rad2deg(np.pi/2))

    # turn bob's measurement plates to measure H/sqrt(2) - (e^i*phi)*V/sqrt(2)
    print(m.time, "Turning Bob's measurement plates")
    m.configure_motors(B_HWP = np.rad2deg(HWP_angle), B_QWP = np.rad2deg(QWP_angle))

    # sweep the QP to determine the minimum count angle
    # sweeps through negative angles so that laser reflection points inward
    m.sweep("C_QP", -30, 0, 25, 5, 3)

    print(m.time, "Sweep complete")

    # read the data into a dataframe
    data = m.close_output()
    
    # shutsdown the manager
    m.shutdown()

    # take the counts of the quartz sweep at each angle and find the index of the minimum data point
    QP_counts = data["C4"]
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
        if data["C_QP"][i] <= new_guess + RANGE:
            max_bound = i
    
    # create new truncated data set using min_bound and max_bound
    fit_data = QP_counts[min_bound:max_bound]
    fit_angles = data["C_QP"][min_bound:max_bound]
    fit_unc = data["C4_sem"][min_bound:max_bound]
    # perform a new sweep around the previous minimum data point
    # m.new_output("QP_sweep2.csv")
    # m.sweep("C_QP", new_guess - RANGE, new_guess + RANGE, 20, 5, 0.1)
    # instead, performed one detailed sweep with data truncated for fitting and analysis

    # refine analysis and fitting since we don't know what the function will look like
    # find the lowest data point and then resweep near the guessed minimum to fit a function
    # write own fit function in analysis maybe

    args1, unc1 = analysis.fit('quartic', fit_angles, fit_data, fit_unc)

    plt.title('Angle of QP to minimize counts')
    plt.xlabel('QP angle (deg))')
    plt.ylabel('Count rates (#/s)')
    plt.errorbar(fit_angles, fit_data, yerr=fit_unc, fmt='o', label="Measurement Basis")
    analysis.plot_func('quartic', args1, fit_angles, label='Measurement fit function')
    plt.legend()
    plt.show()

    # finds the angle that corresponds to the minimum value of the fit function
    def fit(x):

        return args1[0] * x**4 + args1[1] * x**3 + args1[2] * x**2 + args1[3] * x + args1[4]

    minimum = opt.minimize(fit, new_guess)
    min_angle = minimum.pop('x')

    print(min_angle)

    """
    basis preset:
    HWP = b + u / 2 from horizontal
    QWP = u + pi/2 from horizontal

    N vv/sec is the number of vv counts per second, same with HH
    """

def UVHWP_sweep(m:Manager, ratio):
    """
    Adapted from Alec's code in balance.py
    """

    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'

    # sweep in second octant because the standard phi plus state setting for the UV_HWP is there
    GUESS = 67.5
    RANGE = 22.5
    N = 20
    SAMP = (5, 3)

    PCT1 = ratio

    m.new_output("UVHWP_balance_sweep1.csv")

    # configure measurement basis
    print(m.time, f'Configuring measurement basis {BASIS1}')
    m.meas_basis(BASIS1)

    # do sweep
    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS1}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    # obtain the first round of data and switch to a new output file
    data1 = m.close_output()
    m.new_output('UVHWP_balance_sweep_2.csv')

    # sweep in the second basis
    print(m.time, f'Configuring measurement basis {BASIS2}')
    m.meas_basis(BASIS2)

    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS2}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    print(m.time, 'Data collected, shutting down...')
    data2 = m.close_output()
    
    print(m.time, 'Data collection complete and manager shut down, beginning analysis...')
    m.shutdown()

    args1, unc1 = analysis.fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_sem)
    args2, unc2 = analysis.fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_sem)
    x = analysis.find_ratio('sin2_sq', args1, 'sin2_sq', args2, PCT1, data1.C_UV_HWP, GUESS)

    # print result
    print(f'{COMPONENT} angle to find {PCT1*100:.2f}% coincidences ({(1-PCT1)*100:.2f}% coincidences): {x:.5f}')
    
    # plot the data and fit
    plt.title(f'Angle of {COMPONENT} to achieve {PCT1*100:.2f}% {BASIS1}\ncoincidences ({(1-PCT1)*100:.2f}% {BASIS2} coincidences): {x:.5f} degrees')
    plt.xlabel(f'{COMPONENT} angle (deg)')
    plt.ylabel(f'Count rates (#/s)')
    plt.errorbar(data1.C_UV_HWP, data1.C4, yerr=data1.C4_sem, fmt='o', label=BASIS1)
    plt.errorbar(data2.C_UV_HWP, data2.C4, yerr=data2.C4_sem, fmt='o', label=BASIS2)
    analysis.plot_func('sin2_sq', args1, data1.C_UV_HWP, label=f'{BASIS1} fit function')
    analysis.plot_func('sin2_sq', args2, data2.C_UV_HWP, label=f'{BASIS2} fit function')
    plt.legend()
    plt.show()

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_rho(alpha, beta):

    H = ket([1,0])
    V = ket([0,1])

    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)

    psi = np.cos(alpha)*PSI_PLUS + np.exp(1j*beta)*np.sin(alpha)*PSI_MINUS

    rho = psi @ psi.conj().T

    return rho

if __name__ == '__main__':

    # read user input to determine preset angles for state in radians
    alpha = math.pi/6
    # float(input("Alpha = "))
    beta = math.pi/3
    # float(input("Beta = "))

    """
    Error in measurement HWP and QWP angle settings. Double check all angle calculations.
    """

    """
    Note these angles are all in radians, not degrees.
    """

    # meas_HWP_angle, meas_QWP_angle, HH_frac = get_params(alpha, beta)

    # m = Manager()

    # QP_sweep(m,meas_HWP_angle,meas_QWP_angle)

    """
    If the minimum is 0, we need to compare it to other plot to see the counts to see if the counts are the same as another minimum since 0 is always
    a local minimum. If 0 isn't a minimum, we don't need to worry about this.

    this means that for psi_plus, likely will have to manually set the QP to 0 before running the UVHWP sweep

    Comparing full tomography density matrix and theoretical density matrix:
    diagonal elements should be very close to one another, off diagonals will be close but with a smaller magnitude (around 95% of the value), if diagonals are 0
    they might be nonzero
    """
    # m.C_QP.goto(-20)

    # UVHWP_sweep(m, HH_frac)

    rho = get_rho(alpha,beta)

    fidelity = get_fidelity(rho, rho)


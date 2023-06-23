from core import Manager, analysis
import numpy as np
import math
import matplotlib.pyplot as plt


"""
Procedure:
1. run this program to set creation state to phi plus and setup measurement polarizations for alice and bob
2. sweep/use other optimization method to turn quartz plate to minimize counts <- **use a sweep for both QP and HWP**
3. run program to alternate between HH and VV measurements then turn the UVHWP to give the correct rate ratio by sweeping **utilize balance.py for the UVHWP**
4. turn BCHWP to flip H and V

use ratio of sin and cos to create initial guess for UVHWP and QP range to guess phi
check first on alpha = 0 so that the state is psi plus
Check with full tomography at the very end (look into previous scripts)
"""



def QP_sweep(m:Manager, u1, b1):

    # set the output file for manager
    m.new_output("QP_sweep.csv")

    # set the creation state to phi plus
    print(m.time, "Setting creation state to phi plus")
    m.make_state('phi_plus')
    m.log(f'configured phi_plus: {m._config["state_presets"]["phi_plus"]}')

    # turn alice's measurement plates to measure (H+V)/sqrt(2)
    print(m.time, "Turning Alice's measurement plates")
    m.configure_motors(A_HWP = np.rad2deg(np.pi/4), A_QWP = np.rad2deg(np.pi*3/4))

    # turn bob's measurement plates to measure H/sqrt(2) - (e^i*phi)*V/sqrt(2)
    print(m.time, "Turning Bob's measurement plates")
    m.configure_motors(B_HWP = np.rad2deg((b1+u1)/2), B_QWP = np.rad2deg(u1 + np.pi/2))

    # sweep the QP to determine the minimum count angle
    input("Ready to sweep. Press Enter to continue")
    m.sweep("C_QP", 0, 25, 25, 5, 3)

    print(m.time, "Sweep complete")

    # read the data into a dataframe
    data = m.close_output()
    
    # find new method for shutting down in alec's code

    # take the counts of the quartz sweep at each angle and find the index of the minimum data point
    QP_counts = data["C4"]
    min_ind = 0
    for i in range(len(QP_counts)):
        if QP_counts[i] == min(QP_counts):
            min_ind = i
    

    new_guess = data["C_QP"][min_ind]
    RANGE = 5

    min_bound = 0
    max_bound = len(QP_counts)

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

    args1, unc1 = analysis.fit('quadratic', fit_angles, fit_data, fit_unc)

    plt.title('Angle of QP to minimize counts')
    plt.xlabel('QP angle (deg))')
    plt.ylabel('Count rates (#/s)')
    plt.errorbar(fit_angles, fit_data, yerr=fit_unc, fmt='o', label="Measurement Basis")
    analysis.plot_func('quadratic', args1, fit_angles, label='Measurement fit function')
    plt.legend()
    plt.show()

    print(args1)
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

    GUESS = 22.28
    RANGE = 24.0
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

    args1, unc1, = analysis.fit('sin2', data1.C_UV_HWP, data1.C4, data1.C4_sem)
    args2, unc2, = analysis.fit('sin2', data2.C_UV_HWP, data2.C4, data2.C4_sem)
    x = analysis.find_ratio('sin2', args1, 'sin2', args2, PCT1, data1.C_UV_HWP, GUESS)

    # print result
    print(f'{COMPONENT} angle to find {PCT1*100:.2f}% coincidences ({(1-PCT1)*100:.2f}% coincidences): {x:.5f}')
    
    # plot the data and fit
    plt.title(f'Angle of {COMPONENT} to achieve {PCT1*100:.2f}% {BASIS1}\ncoincidences ({(1-PCT1)*100:.2f}% {BASIS2} coincidences): {x:.5f} degrees')
    plt.xlabel(f'{COMPONENT} angle (rad)')
    plt.ylabel(f'Count rates (#/s)')
    plt.errorbar(data1.C_UV_HWP, data1.C4, yerr=data1.C4_sem, fmt='o', label=BASIS1)
    plt.errorbar(data2.C_UV_HWP, data2.C4, yerr=data2.C4_sem, fmt='o', label=BASIS2)
    analysis.plot_func('sin2', args1, data1.C_UV_HWP, label=f'{BASIS1} fit function')
    analysis.plot_func('sin2', args2, data2.C_UV_HWP, label=f'{BASIS2} fit function')
    plt.legend()
    plt.show()

def full_tomography(self):

    """
    Full_tomography: [ "LA", "RA", "VA", "HA", "DA", "AA",
        "AD", "DD", "HD", "VD", "RD",  "LD",
        "LH", "RH", "VH", "HH", "DH", "AH", 
        "AV", "DV", "HV", "VV", "RV", "LV",
        "LL", "RL", "VL", "HL", "DL", "AL",
        "AR", "DR", "HR", "VR", "RR", "LR"]
    """

    for basis in m._config['Full_tomography']:
        m.meas_basis(basis)
        m.take_data()

if __name__ == '__main__':

    # read user input to determine preset angles for state in radians
    alpha = float(input("Alpha = "))
    beta = float(input("Beta = "))

    # calculate phi for different cases of alpha and beta

    if alpha <= math.pi/2 and beta <= math.pi/2:
        r1 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        delta = math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1))
        gamma = math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2))
        phi = gamma + delta

    if alpha >= math.pi/2 and beta >= math.pi/2:
        r1 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        delta = math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1))
        gamma = math.pi + math.asin((math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2))
        phi = gamma - delta

    if alpha <= math.pi/2 and beta >= math.pi/2:
        r1 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        delta = (math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1)
        gamma = (math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2)
        phi = gamma + delta
        
    if alpha >= math.pi/2 and beta <= math.pi/2:
        r1 = math.sqrt(((1-math.sin(2*alpha)*math.cos(beta))/2))
        r2 = math.sqrt(((1+math.sin(2*alpha)*math.cos(beta))/2))
        delta = (math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r1)
        gamma = math.pi + (math.sin(alpha)*math.sin(beta))/(math.sqrt(2)*r2)
        phi = gamma - delta

    # calculate theta based on alpha and beta
    theta = math.sqrt(math.acos((1+math.cos(beta)*math.sin(2*alpha))/2))

    # find angles b and u, which determine the angle Bob's measurement waveplates should be oriented
    b = np.pi/4
    u = phi/2

    meas_HWP_angle = b + u / 2
    meas_QWP_angle = u + math.pi/2

    rate_ratio = (math.tan(theta))**2

    """
    Note these angles are all in radians, not degrees.
    """

    m = Manager()

    # QP_sweep(m,u,b)

    m.C_QP.goto(17.33075)

    UVHWP_sweep(m, rate_ratio)





    

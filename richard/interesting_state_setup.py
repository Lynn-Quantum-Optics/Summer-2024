from core import Manager, analysis
import numpy as np
import math as m


"""
Procedure:
1. run this program to set creation state to phi plus and setup measurement polarizations for alice and bob
2. sweep/use other optimization method to turn quartz plate to minimize counts <- **use a sweep for both QP and HWP**
3. run program to alternate between HH and VV measurements then turn the UVHWP to give the correct rate ratio by sweeping **utilize balance.py for the UVHWP**
4. turn BCHWP to flip H and V

use ratio of sin and cos to create initial guess for UVHWP and QP range to guess phi
Check with full tomography at the very end (look into previous scripts)
"""

# # read user input to determine preset angles for state in radians
# alpha = float(input("Alpha = "))
# beta = float(input("Beta = "))
    
# calculate phi for different cases of alpha and beta

def QP_sweep(m:Manager, alpha, beta):

    if alpha <= m.pi/2 and beta <= m.pi/2:
        r1 = m.sqrt(((1+m.sin(2*alpha)*m.cos(beta))/2))
        r2 = m.sqrt(((1-m.sin(2*alpha)*m.cos(beta))/2))
        delta = m.asin((m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r1))
        gamma = m.asin((m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r2))
        phi = gamma + delta

    if alpha >= m.pi/2 and beta >= m.pi/2:
        r1 = m.sqrt(((1-m.sin(2*alpha)*m.cos(beta))/2))
        r2 = m.sqrt(((1+m.sin(2*alpha)*m.cos(beta))/2))
        delta = m.asin((m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r1))
        gamma = m.pi + m.asin((m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r2))
        phi = gamma - delta

    if alpha <= m.pi/2 and beta >= m.pi/2:
        r1 = m.sqrt(((1+m.sin(2*alpha)*m.cos(beta))/2))
        r2 = m.sqrt(((1-m.sin(2*alpha)*m.cos(beta))/2))
        delta = (m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r1)
        gamma = (m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r2)
        phi = gamma + delta
        
    if alpha >= m.pi/2 and beta <= m.pi/2:
        r1 = m.sqrt(((1-m.sin(2*alpha)*m.cos(beta))/2))
        r2 = m.sqrt(((1+m.sin(2*alpha)*m.cos(beta))/2))
        delta = (m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r1)
        gamma = m.pi + (m.sin(alpha)*m.sin(beta))/(m.sqrt(2)*r2)
        phi = gamma - delta

    # calculate theta based on alpha and beta
    theta = m.sqrt(m.acos((1+m.cos(beta)*m.sin(2*alpha))/2))

    # find angles b and u, which determine the angle Bob's measurement waveplates should be oriented
    b = np.pi/4
    u = phi/2

    HWP_angle = b + u / 2
    QWP_angle = u + m.pi/2

    rate_ratio = (m.tan(theta))**2

    # set the creation state to phi plus
    m.make_state('phi_plus')
    m.log(f'configured phi_plus: {m._config["state_presets"]["phi_plus"]}')

    # turn alice's measurement plates to measure (H+V)/sqrt(2)
    m.configure_motors("A_HWP" = np.rad2deg(np.pi/4), "A_QWP" = np.rad2deg(np.pi*3/4))

    # turn bob's measurement plates to measure H/sqrt(2) - (e^i*phi)*V/sqrt(2)
    m.configure_motors("B_HWP" = np.rad2deg((b+u)/2), "B_QWP" = np.rad2deg(u + np.pi/2)

    # sweep the AP to determine the minimum count angle
    m.sweep("QP", 0, 25, 25, 5, 0.1)

    data = m.close_output()

    m.shutdown()

    args1, unc1, _ = analysis.fit('quadratic', data.QP, data.C4, data.C4_sem)

    plt.title('Angle of QP to minimize counts')
    plt.xlabel('QP angle (rad)')
    plt.ylabel('Count rates (#/s)')
    plt.errorbar(data.QP, data.C4, yerr=data.C4_sem, fmt='o', label="Measurement Basis")
    analysis.plot_func('quadratic', args1, data.QP, label='Measurement fit function')
    plt.legend()
    plt.show()

    """
    basis preset:
    HWP = b + u / 2 from horizontal
    QWP = u + pi/2 from horizontal


    N vv/sec is the number of vv counts per second, same with HH
    """

def UVHWP_sweep(m:Manager, alpha, beta):

    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'
    STATE = 'phi_plus'

    PCT1 = 0.50

    # configure measurement basis
    print(m.time, f'Configuring measurement basis {BASIS1}')
    m.meas_basis(BASIS1)

    # do sweep
    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS1}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    # obtain the first round of data and switch to a new output file
    data1 = m.close_output()
    m.new_output('balance_sweep_2.csv')

    # sweep in the second basis
    print(m.time, f'Configuring measurement basis {BASIS2}')
    m.meas_basis(BASIS2)

    print(m.time, f'Beginning sweep of {COMPONENT} from {GUESS-RANGE} to {GUESS+RANGE}, measurement basis: {BASIS2}')
    m.sweep(COMPONENT, GUESS-RANGE, GUESS+RANGE, N, *SAMP)

    print(m.time, 'Data collected, shutting down...')
    data2 = m.close_output()
    m.shutdown()
    
    print(m.time, 'Data collection complete and manager shut down, beginning analysis...')

    args1, unc1, _ = analysis.fit('sin2', data1.C_UV_HWP, data1.C4, data1.C4_sem)
    args2, unc2, _ = analysis.fit('sin2', data2.C_UV_HWP, data2.C4, data2.C4_sem)
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




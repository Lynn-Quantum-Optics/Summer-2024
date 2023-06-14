# from core import Manager
import numpy as np
import math as m


"""
Procedure:
1. run this program to set creation state to phi plus and setup measurement polarizations for alice and bob
2. sweep/use other optimization method to turn quartz plate to minimize counts
3. run program to alternate between HH and VV measurements then turn the UVHWP to give the correct rate ratio by sweeping
4. turn BCHWP to flip H and V
"""

# read user input to determine preset angles for state in radians
alpha = float(input("Alpha = "))
beta = float(input("Beta = "))
    
# calculate phi for different cases of alpha and beta

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

# if __name__ == '__main__':

#     # intialize the manager
#     m = Manager(out_file='quartz_sweep.csv')


#     # set the creation state to phi plus
#     m.make_state('phi_plus')
#     m.log(f'configured phi_plus: {m._config["state_presets"]["phi_plus"]}')

#     # turn alice's measurement plates to measure (H+V)/sqrt(2)
#     m.configure_motors("A_HWP" = np.rad2deg(np.pi/4), "A_QWP" = np.rad2deg(np.pi*3/4))

#     # turn bob's measurement plates to measure H/sqrt(2) - (e^i*phi)*V/sqrt(2)
#     m.configure_motors("B_HWP" = np.rad2deg((b+u)/2), "B_QWP" = np.rad2deg(u + np.pi/2)


#     """
#     basis preset:
#     HWP = b + u / 2 from horizontal
#     QWP = u + pi/2 from horizontal


#     N vv/sec is the number of vv counts per second, same with HH
#     """





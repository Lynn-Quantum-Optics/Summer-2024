import numpy as np
from core import Manager, analysis

def meas_phi(m:Manager, samp:'tuple[int,float]'):
    m.log('BEGIN SUBROUTINE: meas_phi')

    # get HH and VV count rates
    m.log('Collecting HH counts...')
    m.meas_basis('HH')
    rate_HH, unc_HH = m.take_data(*samp, 'C4')
    m.log('Collecting VV counts...')
    m.meas_basis('VV')
    rate_VV, unc_VV = m.take_data(*samp, 'C4')

    # calculate theta
    theta = np.arccos(np.sqrt(rate_HH/(rate_HH + rate_VV)))
    m.log(f'Calculated theta = {np.rad2deg(theta):.5f} degrees')

    # get DR and DL rates
    m.log('Collecting DR counts...')
    m.meas_basis('DR')
    rate_DR, unc_DR = m.take_data(*samp, 'C4')
    m.log('Collecting DL counts...')
    m.meas_basis('DL')
    rate_DL, unc_DL = m.take_data(*samp, 'C4')
    
    # calculate phi
    phi = np.arcsin((rate_DR-rate_DL)/(rate_DR+rate_DL) * 1/np.sin(2*theta))
    m.log(f'Calculated phi = {np.rad2deg(phi):.5f} degrees')

    return np.rad2deg(theta), np.rad2deg(phi)

def main():
    m = Manager()
    m.make_state('phi_plus')
    
    meas_phi(m, (5,2))

    m.shutdown()

if __name__ == '__main__':
    main()
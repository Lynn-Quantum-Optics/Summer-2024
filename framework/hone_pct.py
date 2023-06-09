import matplotlib.pyplot as plt
from core import Manager, analysis
import numpy as np

def hone_pct(m:Manager, component:str, target_pct_1:float, guess:float, delta:float, num_samp:int, samp_period:float, *bases):
    ''' Hone in on a percentage of detections in a given basis.
    '''
    m.log('BEGIN SUBROUTINE: hone_pct')
    # angles that we will take data at
    xs = [guess - delta, guess, guess + delta]
    m.log(f'Angles to test: {", ".join([str(x) for x in xs])}')
    
    # output coincidence rates, empty lists for each basis
    ys = [[] for _ in bases]
    y_errs = [[] for _ in bases]

    # take data at each angle for each basis. the basis motors are slower so we manipulate them minimally
    for i, b in enumerate(bases):
        m.log(f'Taking data for basis {b}')
        # set measurement basis
        m.meas_basis(b)
        # take data at each angle
        for x in xs:
            m.log(f'Moving {component} to {x}')
            # set the component
            m.configure_motors(**{component:x})
            # take data
            m.log('Taking data...')
            rate, unc = m.take_data(num_samp, samp_period, 'C4')
            # append to list
            ys[i].append(rate)
            y_errs[i].append(unc)
    
    # calculate percentage data based on rates
    ys = np.array(ys)
    y_errs = np.array(y_errs)
    pct_data = ys[0]/np.sum(ys, axis=0)
    # disgusting error expression based on *the math*
    pct_errs = np.sqrt((np.sum(ys[1:], axis=0)*y_errs[0])**2 + ys[0]**2 * np.sum(y_errs**2, axis=0)) / np.sum(ys, axis=0)**2

    # print out some notes
    for i, x in enumerate(xs):
        m.log(f'{component} = {x} deg -> {pct_data[i]*100:.5f} ± {pct_errs[i]*100:.5f}% {bases[0]} detections')
    
    # do a linear regression to find the angle that gives the target percentage
    m.log('Doing linear regression to determine optimal angle')
    (slope,intercept), _, _ = analysis.fit('line', xs, pct_data, pct_errs)
    target_angle = (target_pct_1 - intercept) / slope
    m.log(f'Calculated best angle: {target_angle} deg')

    
    # obtain data for the target angle
    m.log('Taking data for target angle...')
    rates = []
    uncs = []
    m.configure_motors(**{component:target_angle})
    for b in bases:
        m.log(f'Configuring basis {b}')
        m.meas_basis(b)
        rate, unc = m.take_data(num_samp, samp_period, 'C4')
        rates.append(rate)
        uncs.append(unc)

    rates = np.array(rates)
    uncs = np.array(uncs)
    
    m.log(f'Calculating final statistics for output')
    # calculate percentage data based on rates
    pct1 = rates[0]/np.sum(rates)
    # disgusting error expression based on *the math*
    pct1_unc = np.sqrt((np.sum(rates[1:])*uncs[0])**2 + rates[0]**2 * np.sum(uncs[1:]**2)) / np.sum(rates)**2
    
    # calculate angle uncertainty (kinda)
    angle_unc = abs(pct1_unc / slope)
    
    m.log(f'target_angle = {target_angle}')
    m.log(f'angle_unc = {angle_unc}')
    m.log(f'pct1 = {pct1}')
    m.log(f'pct1_unc = {pct1_unc}')

    m.log('END SUBROUTINE: hone_pct')
    return target_angle, angle_unc, pct1, pct1_unc

def main():
    # parameters
    GUESS = 65.86
    RANGE = 0.5
    NUM_SAMP = 5
    SAMP_PERIOD = 1
    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'
    PCT1 = 0.50
    
    # initialize manager
    m = Manager()
    m.log('initialization complete')

    # start with taking data at either end of the range
    m.log(f'configuring phi+')
    m.make_state('phi_plus')

    target_angle, angle_unc, pct1, pct1_unc = hone_pct(m, COMPONENT, PCT1, GUESS, RANGE, NUM_SAMP, SAMP_PERIOD, BASIS1, BASIS2)

    print(target_angle, angle_unc, pct1, pct1_unc)

    m.shutdown()

if __name__ == '__main__':
    main()

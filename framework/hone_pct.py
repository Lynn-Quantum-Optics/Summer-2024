import matplotlib.pyplot as plt
from core import Manager, analysis
import numpy as np

def hone_pct(m:Manager, component:str, target_pct_1:float, guess:float, delta:float, num_samp:int, samp_period:float, within:float, *bases):
    ''' Hone in on a percentage of detections in a given basis.
    '''
    # note routine
    m.log('BEGIN ROUTINE: hone_pct')
    
    # +++ prepare data lists +++
    angles = [] # angles we actually hit
    pct_datas = []
    pct_errs = []

    # +++ define function to collect a single data point +++
    def add_data_point(x):
        m.log(f'Moving {component} to {x}')
        angle = m.configure_motor(component, x)
        angles.append(angle)
        # take data for each basis
        rates = []
        uncs = []
        for i, b in enumerate(bases):
            # set measurement basis
            m.log(f'Setting measurement basis {b}')
            m.meas_basis(b)
            rate, unc = m.take_data(num_samp, samp_period, 'C4')
            # append to list
            rates.append(rate)
            uncs.append(unc)
        # calculate stats
        m.log(f'Calculating statistics for this position')
        rates = np.array(rates)
        uncs = np.array(uncs)
        # calculate percentage data based on rates
        pct1 = rates[0]/np.sum(rates)
        # disgusting error expression based on *the math*
        pct1_unc = np.sqrt((np.sum(rates[1:])*uncs[0])**2 + rates[0]**2 * np.sum(uncs[1:]**2)) / np.sum(rates)**2
        # calculate angle uncertainty (kinda)
        m.log(f'angle = {angle:.5f} degrees')
        m.log(f'pct1 = {pct1*100:.5f}%')
        m.log(f'pct1_unc = {pct1_unc*100:.5f}%')
        # add the data to the global lists
        pct_datas.append(pct1)
        pct_errs.append(pct1_unc)
        return angle, pct1, pct1_unc
    
    # +++ start with initial "sweep" +++

    # angles that we will take data at to start
    xs_t = [guess - delta, guess, guess + delta] # xs that we *want* to hit
    m.log(f'Initial angles to test: {", ".join([str(x) for x in xs_t])}')

    # take data for each position of the motor
    for x in xs_t:
        add_data_point(x)

    # finished phase 1, print a note
    m.log('Done with initial sweep.')
    
    # loop to go until we get close enough
    m.log('Moving to iterative regressions.')
    loop_count = 0
    while loop_count < 10:
        m.log(f'Iterative regression, round {loop_count+1}.')
        # do a linear regression to find the angle that gives the target percentage
        (slope,intercept), _, _ = analysis.fit('line', angles, pct_datas, pct_errs)
        target_angle = (target_pct_1 - intercept) / slope
        m.log(f'Calculated best angle: {target_angle} deg')
        
        # obtain data for the target angle
        m.log('Taking data at target angle...')
        ang, pct, unc = add_data_point(target_angle)

        # check if it's good enough
        if abs(target_pct_1 - pct) / unc < within:
            m.log(f'pct1 is {abs(target_pct_1 - pct)/unc:.3f} sigma from target (less than requested {within}) -> terminating search')
            break
        else:
            # not good enough, try again
            loop_count += 1
            continue

    m.log('END ROUTINE: hone_pct')
    return angles[-1], pct_datas[-1], pct_errs[-1]

def min_det(m:Manager, component:str, basis:str, guess:float, delta:float, num_sweep:int, samp:'tuple[int,float]'):
    m.log('BEGIN ROUTINE: min_det')
    
    # initialize data lists
    x = []
    y = []
    yerr = []

    # configure measurement
    m.log(f'Configuring measurement basis: {basis}')
    m.meas_basis(basis)

    # loop to sweep
    for angle in np.linspace(guess-delta, guess+delta, num_sweep):
        # move component
        m.log(f'Moving {component} to {angle}')
        actual_angle = m.configure_motor(component, angle)
        x.append(actual_angle)

        # take datas
        rate, unc = m.take_data(*samp, 'C4')
        y.append(rate)
        yerr.append(unc)
    
    # fit a quadratic expression


    m.log('END ROUTINE: min_det')
    return theta

def main():
    # parameters
    GUESS = 65.86
    RANGE = 0.5
    NUM_SAMP = 5
    SAMP_PERIOD = 3
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

    target_angle, pct1, pct1_unc = hone_pct(m, COMPONENT, PCT1, GUESS, RANGE, NUM_SAMP, SAMP_PERIOD, 0.4, BASIS1, BASIS2)

    print(target_angle, pct1, pct1_unc)

    m.shutdown()

if __name__ == '__main__':
    main()

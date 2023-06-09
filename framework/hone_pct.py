import matplotlib.pyplot as plt
from core import Manager, analysis
import numpy as np

def hone_pct(m:Manager, component:str, target_pct_1:float, guess:float, delta:float, num_samp:int, samp_period:float, *bases):
    ''' Hone in on a percentage of detections in a given basis.
    '''
    # note routine
    m.log('BEGIN ROUTINE: hone_pct')
    
    # angles that we will take data at to start
    xs_ = [guess - delta, guess, guess + delta] # xs that we *want* to hit
    m.log(f'Initial angles to test: {", ".join([str(x) for x in xs_])}')
    
    # output coincidence rates, empty lists for each basis
    xs = [[] for _ in bases] # actual angles we hit
    ys = [[] for _ in bases]
    y_errs = [[] for _ in bases]

    # take data at each angle for each basis. the basis motors are slower so we manipulate them minimally
    for i, b in enumerate(bases):
        m.log(f'Setting measurement basis {b}')
        # set measurement basis
        m.meas_basis(b)
        # take data at each angle
        for x in xs_:
            m.log(f'Moving {component} to {x}')
            # set the component
            xs.append(m.configure_motor(component, x))
            rate, unc = m.take_data(num_samp, samp_period, 'C4')
            # append to list
            ys[i].append(rate)
            y_errs[i].append(unc)
    
    # finished phase 1, print a note
    m.log('Done with initial sweep.')
    
    # calculate percentage data based on rates
    ys = np.array(ys)
    y_errs = np.array(y_errs)
    pct_data = ys[0]/np.sum(ys, axis=0)
    # disgusting error expression based on *the math*
    pct_errs = np.sqrt((np.sum(ys[1:], axis=0)*y_errs[0])**2 + ys[0]**2 * np.sum(y_errs**2, axis=0)) / np.sum(ys, axis=0)**2

    # print out some notes
    for i, x in enumerate(xs_):
        m.log(f'{component} = {x} deg -> {pct_data[i]*100:.5f} ± {pct_errs[i]*100:.5f}% {bases[0]} detections')
    
    # loop to go until we get close enough
    m.log('Moving to iterative regressions.')
    loop_count = 0
    while loop_count < 10:
        m.log(f'Iterative regression, round {loop_count+1}.')
        # do a linear regression to find the angle that gives the target percentage
        (slope,intercept), _, _ = analysis.fit('line', xs_, pct_data, pct_errs)
        target_angle = (target_pct_1 - intercept) / slope
        m.log(f'Calculated best angle: {target_angle} deg')
        
        # obtain data for the target angle
        m.log('Taking data at target angle...')
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
        
        m.log(f'Calculating statistics for output')
        # calculate percentage data based on rates
        pct1 = rates[0]/np.sum(rates)
        # disgusting error expression based on *the math*
        pct1_unc = np.sqrt((np.sum(rates[1:])*uncs[0])**2 + rates[0]**2 * np.sum(uncs[1:]**2)) / np.sum(rates)**2
        # calculate angle uncertainty (kinda)
        angle_unc = abs(pct1_unc / slope)
        # log the output values
        m.log(f'target_angle = {target_angle}')
        m.log(f'angle_unc = {angle_unc}')
        m.log(f'pct1 = {pct1}')
        m.log(f'pct1_unc = {pct1_unc}')
        
        # check if it's good enough
        if abs(target_pct_1 - pct1) < pct1_unc:
            m.log(f'pct1 is {abs(target_pct_1 - pct1)/pct1_unc:.3f} sigma from target -> terminating search')
            break
        else:
            # not good enough, add this datapoint and try again
            xs_ = np.append(xs_, [target_angle], axis=0)
            pct_data = np.append(pct_data, [pct1], axis=0)
            pct_errs = np.append(pct_errs, [pct1_unc], axis=0)
            continue

    m.log('END ROUTINE: hone_pct')
    return target_angle, angle_unc, pct1, pct1_unc

def min_det(m:Manager, component:str, guess:float, delta:float, num_sweep:int, samp:'tuple[int,float]'):
    m.log('BEGIN ROUTINE: min_det')
    x = []
    y = []
    yerr = []


    for angle in np.linspace(guess-delta, guess+delta, num_sweep):


    return theta
    m.log('END ROUTINE: min_det')

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

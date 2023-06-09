import matplotlib.pyplot as plt
from core import Manager, analysis
import numpy as np

def hone_pct(m:Manager, component:str, target_pct_1:float, guess:float, delta:float, num_samp:int, samp_period:float, *bases):
    ''' Hone in on a percentage of detections in a given basis.
    '''
    # angles that we will take data at
    xs = [guess - delta, guess, guess + delta]
    # list of empty lists for each basis
    ys = [[]]*len(bases)
    y_errs = [[]]*len(bases)

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
            rate, unc = m.take_data(num_samp, samp_period, 'C4')
            # append to list
            ys[i].append(pct)
            y_errs[i].append(unc)
    
    # calculate percentage data based on rates
    ys = np.array(ys)
    y_errs = np.array(y_errs)
    pct_data = ys[0]/np.sum(ys, axis=0)
    # disgusting error expression based on *the math*
    pct_errs = np.sqrt((np.sum(ys[1:], axis=0)*y_errs[0])**2 + ys[0]**2 * np.sum(y_errs**2, axis=0)) / np.sum(ys, axis=0)**2

    # print out some notes
    for i, x in enumerate(xs):
        m.log(f'{component} = {x} deg --> {pct_data[i]*100:.5f} ± {pct_errs[i]*100:.5f}%')
    
    # do a linear regression to find the angle that gives the target percentage
    (slope,intercept), _, _ = analysis.fit('line', xs, pct_data, pct_errs)
    target_angle = (target_pct_1 - intercept) / slope

    # obtain data for the target angle
    m.configure_motors(**{component:target_angle})
    
    rates = []
    uncs = []

    for b in bases:
        m.meas_basis(b)
        rate, unc = m.take_data(num_samp, samp_period, 'C4')
        rates.append(rate)
        uncs.append(unc)
    
    # calculate percentage data based on rates
    pct1 = rates[0]/np.sum(rates)
    # disgusting error expression based on *the math*
    pct1_unc = np.sqrt((np.sum(rates[1:])*uncs[0])**2 + rates[0]**2 * np.sum(uncs[1:]**2))
    
    # calculate angle uncertainty (kinda)
    angle_unc = pct1_unc / slope
    
    return target_angle, angle_unc, pct1, pct1_unc






if __name__ == '__main__':
    
    # parameters
    GUESS = 65.86
    RANGE = 0.5
    COARSE_SAMP = (5, 1)
    FINE_SAMP = (10,5)
    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'
    STATE = 'phi_plus'

    PCT1 = 0.50
    N_MAX = 10
    
    # initialize manager
    m = Manager()
    print(m.time, 'Manager initialized')

    # start with taking data at either end of the range

    
    # configure starting state
    print(m.time, f'Configuring {STATE} state')
    m.make_state(STATE)

    # measure at either end of the range in both basis
    range_angles = [GUESS - RANGE, GUESS + RANGE]
    range_pcts = []
    range_uncs = []
    print(m.time, f'Taking initialization data {COMPONENT} @ {range_angles[0]:.5f} deg')
    m.configure_motors(**{COMPONENT:range_angles[0]}) 
    a, b = m.pct_det(BASIS1, BASIS2, *COARSE_SAMP, 'C4')
    range_pcts.append(a)
    range_uncs.append(b)

    print(m.time, f'Taking initialization data: {COMPONENT} @ {range_angles[1]:.5f} deg')
    m.configure_motors(**{COMPONENT:range_angles[1]}) 
    a, b = m.pct_det(BASIS1, BASIS2, *COARSE_SAMP, 'C4')
    range_pcts.append(a)
    range_uncs.append(b)

    # print the inital range rates and assert
    print(m.time, f'At {range_angles[0]:.3f} deg -> {100*range_pcts[0]}% {BASIS1} detections')
    print(m.time, f'At {range_angles[1]:.3f} deg -> {100*range_pcts[1]}% {BASIS1} detections')

    # initialize data
    all_angles = [] + range_angles
    all_pcts = [] + range_pcts
    all_uncs = [] + range_uncs

    fig = plt.figure()
    ax = fig.add_subplot(111)
    done = False
    for i in range(N_MAX + 1):
        # update plot
        ax.cla()
        ax.set_xlabel(f'{COMPONENT} angle (degrees)')
        ax.set_ylabel(f'Percent of {BASIS1} detections (of {BASIS1} + {BASIS2})')
        ax.errorbar(all_angles, all_pcts, all_uncs, fmt='ro', elinewidth=2, ms=0)
        plt.draw()
        plt.pause(0.01)
        if i == N_MAX:
            print(m.time, 'hit maximum iterations')
            break
        if done:
            break

        # narrow the range using a linear approximate guess
        slope = (range_pcts[1] - range_pcts[0])/(range_angles[1] - range_angles[0])
        new_guess = (PCT1 - range_pcts[0])/slope + range_angles[0]

        print(m.time, f'{i}: new best guess = {new_guess:.5f} degrees')

        # take the data for the new guess
        m.configure_motors(**{COMPONENT:new_guess})
        pct, unc = m.pct_det(BASIS1, BASIS2, *FINE_SAMP, 'C4')
        print(m.time, f'PCT at guess = {pct*100:.5f} +- {unc*100:.7f} %')
        
        # check if there was a problem
        if pct > max(range_pcts) or pct < min(range_pcts):
            print(m.time, 'pct of last guess is outside of the range of pcts. breaking.')
            done = True
        # check if it's as close as we'll get
        if abs(pct-PCT1) < unc/4:
            done = True
            print('Last guess as close as we can get reliably.')

        # add to data lists
        all_pcts.append(pct)
        all_uncs.append(unc)
        all_angles.append(new_guess)
        
        # update the range to search
        if (range_pcts[0] < PCT1 and pct < PCT1) or (range_pcts[0] > PCT1 and pct > PCT1):
            range_angles[0] = new_guess
            range_pcts[0] = pct
        elif (range_pcts[1] < PCT1 and pct < PCT1) or (range_pcts[1] > PCT1 and pct > PCT1):
            range_angles[1] = new_guess
            range_pcts[1] = pct
        else:
            pass
        
        print(m.time, f'{i}: new range angles = {", ".join([f"{a:.5f}" for a in range_angles])}')
    plt.ioff()
    plt.show()

    # for i in range(15):
    #     print(m.time, f'On iteration {i}\n\tCurrent range: {range_angles}\n\tCurrent pcts: {range_pct}')
        
    '''

    # setup the state configuration
    print(m.time, f'Configuring {STATE} state')
    m.make_state(STATE)

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
    x = analysis.find_ratio('sin', args1, 'sin', args2, PCT1, data1.C_UV_HWP, GUESS)

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
    '''

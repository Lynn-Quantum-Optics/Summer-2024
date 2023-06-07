import matplotlib.pyplot as plt
from core import Manager, analysis
import pandas as pd

if __name__ == '__main__':
    
    # parameters
    GUESS = 65.86
    RANGE = 0.5
    SAMP = (5,5)
    COMPONENT = 'C_UV_HWP'
    BASIS1 = 'HH'
    BASIS2 = 'VV'
    STATE = 'phi_plus'

    PCT1 = 0.50
    N_MAX = 10
    
    # initialize manager
    m = Manager()
    print(m.time, 'Manager initialized')

    # configure starting state
    print(m.time, f'Configuring {STATE} state')
    m.make_state(STATE)

    # function to efficiently take data at both ends of a range
    def get_range_data(angle1, angle2):
        m.meas_basis(BASIS1)
        m.configure_motors(**{COMPONENT:angle1})
        

        m.configure_motors(**{COMPONENT:angle2})

    # start with taking data at either end of the range
    
    

    # measure at either end of the range in both basis
    range_angles = [GUESS - RANGE, GUESS + RANGE]
    range_pcts = []
    range_uncs = []
    print(m.time, f'Taking initialization data {COMPONENT} @ {range_angles[0]:.5f} deg')
    m.configure_motors(**{COMPONENT:range_angles[0]}) 
    a, b = m.pct_det(BASIS1, BASIS2, *SAMP, 'C4')
    range_pcts.append(a)
    range_uncs.append(b)

    print(m.time, f'Taking initialization data: {COMPONENT} @ {range_angles[1]:.5f} deg')
    m.configure_motors(**{COMPONENT:range_angles[1]}) 
    a, b = m.pct_det(BASIS1, BASIS2, *SAMP, 'C4')
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
        pct, unc = m.pct_det(BASIS1, BASIS2, *SAMP, 'C4')
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

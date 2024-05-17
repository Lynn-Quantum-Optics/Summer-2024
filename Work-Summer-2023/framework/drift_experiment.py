from core import Manager
import time
import pandas as pd

if __name__ == '__main__':
    # parameters
    SAMP = (6, 5)
    TOTAL_TIME = 60 * 3 # collect 3 hours of data

    # initialize manager
    m = Manager(out_file='drift_experiment_all_data.csv')
    m.log('manager initialization complete')

    # setup the state
    m.make_state('phi_plus')
    m.log(f'configured phi_plus: {m._config["state_presets"]["phi_plus"]}')

    # get ready for main loop -- remember start time
    minute = -1
    start_time = time.time()
    m.log(f'experiment initialized, starting main loop at time {start_time} ({m.now})')
    
    # main loopski!
    while minute < TOTAL_TIME:
        # check if too much time has passed (more than a minute)
        if ((time.time() - start_time) // 60) > (minute + 1):
            m.log('PROBLEM ENCOUNTERED: Last round took more than a minute. Data for an intermediate minute will have been skipped.')
        # time for the next test?
        if ((time.time() - start_time) // 60) > minute:
            # log experiment time
            minute += 1
            m.log(f'Starting minute {minute} at {m.now}')

            # select basis
            if minute % 2 == 0:
                m.log('Taking data in HH')
                m.meas_basis('HH')
            elif minute % 2 == 1:
                m.log('Taking data in VV')
                m.meas_basis('VV')
            
            # take data
            m.log('Taking data...')
            rate, unc = m.take_data(*SAMP, 'C4')
            m.log(f'Finished minute {minute} at {m.now}')
        # take a lil nap after every iteration
        time.sleep(0.1)
    
    # done with main loop! save data
    m.log('Main loop complete, saving data...')
    m.close_output(get_data=False)

    # shutdown
    m.log('Drift experiment complete! Shutting down.')
    m.shutdown()


        

         



from core import Manager
import time
import pandas as pd

if __name__ == '__main__':
    # parameters
    SAMP = (6, 5)
    TOTAL_TIME = 60 * 4 # collect 4 hours of data

    # initialize manager
    m = Manager(out_file='drift_experiment_all_data.csv')
    m.log('manager initialization complete')

    # setup the state
    m.make_state('phi_plus')
    m.log(f'configured phi_plus: {m._config["state_presets"]["phi_plus"]}')

    # initialize dataframe dictionary
    df = {
        'time (min)':[],
        'HH':[],
        'VV':[],
        'HH_err':[],
        'VV_err':[]}

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
            m.log(f'Starting minute {minute} at {m.now})')

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

            # adding data to dataframe
            m.log('Appending to dataframe...')
            
            df['time (min)'].append(minute)
            if minute % 2 == 0:
                # took data in HH
                df['HH'].append(rate)
                df['HH_err'].append(unc)
                df['VV'].append(None)
                df['VV_err'].append(None)
            elif minute % 2 == 1:
                # took data in VV
                df['HH'].append(None)
                df['HH_err'].append(None)
                df['VV'].append(rate)
                df['VV_err'].append(unc)

            m.log(f'Finished minute {minute} at {m.now}')
        # take a lil nap after every iteration
        time.sleep(0.1)
    
    # done with main loop! save data
    m.log('Main loop complete, saving data...')
    m.close_output(get_data=False)
    df = pd.DataFrame(df)
    df.to_csv('drift_experiment.csv')

    # shutdown
    m.log('Drift experiment complete! Shutting down.')
    m.shutdown()


        

         



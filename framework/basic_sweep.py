''' basic_sweep.py

This file performs a basic sweep of one optical component through a range of angles, collecting data at even intervals.
'''

from core import Manager
import matplotlib.pyplot as plt
import numpy as np

# this condition is REQUIRED for the multiprocessing that allows for the CCU to run parallel plotting with data collection
if __name__ == '__main__':
    # create a manager object
    m = Manager(out_file='basic_sweep_data.csv')

    # configure DA measurement
    m.meas_basis('HV')

    # loop to collect data
    for i, k in enumerate(np.linspace(-np.pi/3, np.pi/3, 30)):
        print('Iteration', i+1, '/8')
        m.take_data(5,2)
    
    m.shutdown()

    




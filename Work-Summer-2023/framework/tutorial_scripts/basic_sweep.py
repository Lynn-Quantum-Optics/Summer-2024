''' basic_sweep.py

This file performs a basic sweep of one optical component through a range of angles, collecting data at even intervals.
'''

from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np

# this condition is REQUIRED for the multiprocessing that allows for the CCU to run parallel plotting with data collection
if __name__ == '__main__':
    # create a manager object
    m = Manager(config='C:/Users/lynnlab/Documents/Summer-2023/framework/config.json')

    # configure measurement basis and state preset
    m.make_state('phi_plus')
    m.meas_basis('HV')

    # loop to collect data
    for i, angle in enumerate(np.linspace(-60, 60, 30)):
        print('Beginning iteration', i+1, '/30')
        m.B_C_HWP.goto(angle)
        m.take_data(5,0.5)
    
    df = m.output_data('basic_sweep_data.csv')

    m.shutdown()

    # fit the data to a function
    params = analysis.fit('sin2_sq', df['B_C_HWP'], df['C4'])
    
    # plot the function
    analysis.plot_func('sin2_sq', params, df['B_C_HWP'], label=f'${params[0].n:.3f}\\sin^2(2x + {params[1].n:.3f}) + {params[2].n:.3f}$')

    # plot the data
    analysis.plot_errorbars(df['B_C_HWP'], df['C4'], fmt='o', ms=0.1, label='Data')
    
    # setup plot labels and such
    plt.xlabel('Bob\'s Creation HWP Angle')
    plt.ylabel('HV Coincidence Count Rates (#/s)')
    plt.title('Sweep of Bob\'s Creation HWP Angle (degrees)')
    plt.legend()
    
    # save and show the plot
    plt.savefig('basic_sweep_plot.png', dpi=600)
    plt.show()

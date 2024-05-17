from core import Manager
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

if __name__ == '__main__':
    '''
    # initialize the manager in debug mode (no need for count rates)
    m = Manager(debug=True)
    m.init_motors()

    # parameters for test
    motor = m.C_UV_HWP
    target = 65.89814
    N = 50

    # loop to collect data
    errors = []
    for i in tqdm(range(N)):
        # home motor on every run
        m.C_UV_HWP.hardware_home()
        # move and get error
        pos = m.C_UV_HWP.goto(target)
        errors.append(pos - target)

    # write the output to a text file    
    with open('errors.txt', 'w+') as f:
        f.writelines([f'{x}\n' for x in errors])
    '''
    with open('errors.txt', 'r') as f:
        errors = [float(x) for x in f.readlines()]
    av = np.mean(errors)
    sem = stats.sem(errors)
    plt.hist(errors, bins=30)
    # plt.title(f'Histogram of {N} errors for {motor.name}\nmoving to {target} degrees.\nAvg = {av}, SEM = {sem}')
    plt.title(f'Histogram of {50} errors for {"C_UV_HWP"}\nmoving to {65.89814} degrees.\nAvg = {av}, SEM = {sem}')
    plt.xlabel('Error from set point (degrees)')
    plt.ylabel('Frequency')
    plt.show()

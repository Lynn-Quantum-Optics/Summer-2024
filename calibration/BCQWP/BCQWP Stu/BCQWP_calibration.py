from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    TRIAL = 1
    SWEEP_PARAMS = [-2, 2, 15, 5, 3]

    # check for two minimums, one at 0, one at pi plus our offset
    offset = -30
    angle_1 = offset
    angle_2 = 180 + offset

    # initializing manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'BCQWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # make the vv state
    m.make_state('VV')

    # make alice measure in V
    m.log('Sending AQWP to 0')
    m.A_QWP.goto(0)
    m.log('Sending AHWP to 45')
    m.A_HWP.goto(45)

    # Bob to measure in calibrated settings
    m.log('Sending BHWP to 0')
    m.B_HWP.goto(0)
    m.log('Sending BQWP to 0')
    m.B_QWP.goto(0)
    m.log('Sending BCHWP to 0')
    m.B_C_HWP.goto(0)

    # sweep Bob's BCQWP
    m.log("Beginning Bob's BCQWP sweep")

    # manual sweep
    angles, rates = m.sweep('B_C_QWP', *SWEEP_PARAMS)

    


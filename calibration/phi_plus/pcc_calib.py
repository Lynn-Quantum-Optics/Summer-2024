from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp

if __name__ == '__main__':
    SWEEP_PARAMS = [-8, 8, 20, 5, 1]

    # initialize the manager
    m = Manager('../config.json')

    # setup the superposition state
    m.C_UV_HWP.goto(22.5)
    m.C_QP.goto(0)
    m.C_PCC.goto(4.005) # ballpark based on an old calibration
    m.B_C_HWP.goto(0)
from lab_framework import analysis
import uncertainties.unumpy as unp
import numpy as np
import matplotlib.pyplot as plt

def poly5(x, x0, y0, a, b, c, d, e):
    return a*(x-x0)**5 + b*(x-x0)**4 + c*(x-x0)**3 + d*(x-x0)**2 + e*(x-x0) + y0

# params = np.array([15.119901840578258, -0.023645589574515274, -4.64352844174522e-08, -7.833097262301589e-06, -0.00048800601695201746, -0.009672382034741866, -0.06245735180869481])
params = np.array([-9.933800864787777, -157.47365840259604, -0.00015343823276400888, -0.012348721372425101, -0.3948562137567598, -6.266775277416394, -49.56237739150353])

xs = (-25.8-3, -25.8+3)

def make_phase(phi):
    '''
    Returns QP setting, UVHWP offset
    '''
    # try to find in bounds of original fit
    x = analysis.find_value(poly5, params, phi, (-35, 0))
    # check match
    if poly5(x, *params) - phi < 1e-10:
        return x, 0
    # otherwise, find pi+phi
    x = analysis.find_value(poly5, params, np.pi+phi, (-35, 0))
    if poly5(x, *params) - (np.pi + phi) < 1e-10:
        return x, 45
    
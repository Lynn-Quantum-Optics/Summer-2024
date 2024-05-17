'''


'''

# target parameters
STATE_NAME = 'phi_plus'
ALPHA = 45
BETA = 0
PHI = 0

CONFIG_PATH = 'fits.json'

# imports
import json
import numpy as np
# from core import Manager
from core import analysis

# load config dictionary
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# functions

def guess_qp(target_phi):
    ''' Guess the quartz plate angle that achieves a target phi parameter using the config fit. '''
    return analysis.find_value('sec', CONFIG['phi']['params'], target_phi)

def guess_uvhwp(target_alpha):
    ''' Guess the UVHWP angle that achieves a target alpha parameter using the config fit. '''
    # check bounds
    if target_alpha > CONFIG['alpha']['alpha_max']:
        # requesting a target alpha > maximum -> use the maximum
        return CONFIG['alpha']['UVHWP_max']
    elif target_alpha < CONFIG['alpha']['alpha_min']:
        # requesting a target alpha < minimum -> use the minimum
        return CONFIG['alpha']['UVHWP_min']
    else:
        # otherwise use the linear fit
        return analysis.find_value('line', CONFIG['alpha']['params'], target_alpha)

def guess_bchwp(target_beta):
    ''' Guess the UVHWP angle that achieves a target beta parameter using the config fit. '''
    # check bounds
    if target_beta > CONFIG['beta']['beta_max']:
        # requesting a target beta > maximum -> use the maximum
        return CONFIG['beta']['UVHWP_max']
    elif target_beta < CONFIG['beta']['alpha_min']:
        # requesting a target alpha < minimum -> use the minimum
        return CONFIG['beta']['UVHWP_min']
    else:
        # otherwise, we are in the linear-ish region so use the linear fit
        return analysis.find_value('line', CONFIG['beta']['params'], target_beta)

if __name__ == '__main__':
    print(guess_qp(PHI))

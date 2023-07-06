# file to sweep QP for HH and VV settings to determine VV / HH as a function of QP angle #

from core import Manager

m = Manager()

# define sweep params
pos_min = 0
pos_max = 38.3
num_steps = 50
num_sam = 5
samp_period = 1 # seconds

# do HH
m.new_output('HH.csv')
m.make_state('HH')
m.sweep(component='C_QP', pos_min=pos_min, pos_max=pos_max, num_steps=num_steps, num_sam=num_sam, samp_period=samp_period)

m.close_output()

# do VV
m.new_output('VV.csv')
m.make_state('VV')
m.sweep(component='C_QP', pos_min=pos_min, pos_max=pos_max, num_steps=num_steps, num_sam=num_sam, samp_period=samp_period)

m.close_output()

m.shutdown()
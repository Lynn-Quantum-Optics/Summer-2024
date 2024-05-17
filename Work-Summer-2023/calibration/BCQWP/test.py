from lab_framework import Manager
import numpy as np
import time

if __name__ == '__main__':
    m = Manager('../config.json', debug=True)
    m.init_motors()

    for a in np.linspace(-12, 8, 20):
        m.B_C_QWP.goto(a)
        time.sleep(0.25)
    
    m.B_C_QWP.goto(0)
    m.shutdown()


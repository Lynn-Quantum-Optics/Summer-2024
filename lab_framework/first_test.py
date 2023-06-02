import manager
import numpy as np

m = manager.Manager()

N = 20
for i, angle in enumerate(np.linspace(0,np.pi, N)):
    print(f'Iteration {i+1}/{N}: {angle}')
    m.C_HWP.rotate_absolute(angle)
    m.take_data(5, 1)

m.close()

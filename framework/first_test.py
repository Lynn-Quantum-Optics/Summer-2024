import manager
import numpy as np
from tqdm import tqdm



if __name__ == '__main__':

    m = manager.Manager()
    # configure DA measurement
    m.A_HWP.rotate_absolute(np.deg2rad(66.4))
    m.A_QWP.rotate_absolute(np.deg2rad(0.46))
    m.B_HWP.rotate_absolute(np.deg2rad(16.38))
    m.B_QWP.rotate_absolute(np.deg2rad(97.08))
    
    # set UVHWP and 
    m.C_UV_HWP.rotate_absolute(np.deg2rad(22.3+45))
    
    N = 20
    for i, angle in tqdm(enumerate(np.linspace(-0.5,-0.35, N))):
        print(f'Iteration {i+1}/{N}: {angle}')
        m.C_QP.rotate_absolute(angle)
        m.take_data(20, 1)

    m.close()

# PHI PLUS
# C_QP = -0.421
# C_UV_HWP = 1.1746065865921838

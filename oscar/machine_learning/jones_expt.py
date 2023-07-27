# file to extract phi value from experimental matrices using Jones decomp

from jones import *
from rho_methods import *

DATA_PATH = '../../framework/decomp_test/'

def find_phi(filename):
    expt_state = np.load(DATA_PATH+filename)
    rho_expt = expt_state[0]
    
    angles = expt_state[-1]
    UV_HWP_rot, QP_rot, B_C_HWP_rot = angles

    def guess_state(phi):
        '''Given input params, returns the density matrix'''
        def uk_QP():
            return np.array([[1, 0], [0, np.exp(1j*phi)]])
        U = np.kron(np.eye(2), H(B_C_HWP_rot)) @ BBO_expt(phi) @ uk_QP() @ H(UV_HWP_rot)
        return U
    
    def loss(phi):
        1 - np.sqrt(get_fidelity(rho_expt, guess_state(phi)))

    # do GD on params; see how well they are constant as we vary 
    def get_random():
        return np.random.rand()*2*np.pi

    best_phi = get_random()
    phi = best_phi
    best_loss = loss(best_phi)
    n = 0
    num_since_best = 0
    N=1000
    lr = .7
    f = .1
    loss_lim = 1e-3
    while n < N and best_loss > loss_lim:
        result = minimize(loss, phi)
        new_loss = result.fun
        phi = result.x
        if new_loss < best_loss:
            best_loss = new_loss
            best_phi = phi
            num_since_best = 0
        else:
            num_since_best += 1
        if num_since_best > f*N:
            phi = get_random()
        grad = approx_fprime(phi, loss, 1e-8)
        phi -= lr*grad
        n += 1
    return best_phi


## compute phi for all files; fit to determine dependence on UVHWP first since QP = 0, then on QP



    

    
    

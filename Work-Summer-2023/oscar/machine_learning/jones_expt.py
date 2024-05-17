# file to extract phi value from experimental matrices using Jones decomp

from jones import *
from rho_methods import *
import os

DATA_PATH = '../../framework/decomp_test/'

def find_phi(filename):
    expt_state = np.load(DATA_PATH+filename, allow_pickle=True)
    rho_expt = expt_state[0]
    
    angles = expt_state[-1]
    angles = np.deg2rad(angles)
    UV_HWP_rot, QP_rot, B_C_HWP_rot = angles
    print(angles)

    def guess_state(phi):
        '''Given input params, returns the density matrix'''
        def uk_QP():
            return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype='complex')
        U = np.kron(np.eye(2), H(B_C_HWP_rot)) @ BBO_expt(QP_rot) @ uk_QP() @ H(UV_HWP_rot)

        P = U @ s0
        rho = P @ adjoint(P)
        rho/= np.trace(rho)
        return rho
    
    def loss(phi):
        '''Returns the loss function for a given phi'''
        return 1 - np.sqrt(get_fidelity(rho_expt, guess_state(phi)))

    # do GD on params; see how well they are constant as we vary 
    def get_random():
        return np.random.rand()*2*np.pi

    best_phi = get_random()
    phi = best_phi
    best_loss = loss(best_phi)
    new_loss = best_loss
    best_fidelity= get_fidelity(rho_expt, guess_state(best_phi))
    fidelity = best_fidelity
    n = 0
    num_since_best = 0
    N=1000
    lr = 1e-2
    f = .1
    fidelity_targ = .999
    while n < N and best_fidelity < fidelity_targ:
        # print updates
        print('n: ', n, 'fidelity: ', fidelity, 'best phi: ', best_phi)
        print(guess_state(best_phi))
        print('------')
        print(rho_expt)
        result = minimize(loss, phi, bounds=[(0, 2*np.pi)])
        new_loss = result.fun
        phi = result.x
        fidelity = get_fidelity(rho_expt, guess_state(phi))
        if fidelity > best_fidelity:
            best_loss = new_loss
            best_fidelity = fidelity
            best_phi = phi
            num_since_best = 0
        else:
            num_since_best += 1
        if num_since_best > f*N:
            phi = get_random()
        else:
            grad = approx_fprime(phi, loss, 1e-8)
            phi -= lr*grad
        n += 1
    return best_phi, best_fidelity


## compute phi for all files; fit to determine dependence on UVHWP first since QP = 0, then on QP
def det_QP_phi():
    phi_ls = []
    fidelity_ls = []
    qp_ls = np.linspace(-35, 0, 6)
    for qp in qp_ls:
        file = f'rho_qp_{qp}.npy'
        fp = find_phi(file)
        phi = np.rad2deg(fp[0])
        if phi < 0:
            phi += 360
        phi_ls.append(phi)
        fidelity_ls.append(fp[1])

    # save csv
    qp_phi = pd.DataFrame({'QP': qp_ls, 'phi': phi_ls, 'fidelity': fidelity_ls})
    qp_phi.to_csv(join(DATA_PATH, 'qp_phi_reconstr.csv'), index=False)

    # qp_phi = pd.read_csv(join(DATA_PATH, 'qp_phi_reconstr.csv'))
    # qp_ls = qp_phi['QP']
    # phi_ls = qp_phi['phi']
    # fidelity_ls = qp_phi['fidelity']

    # plot phi vs UVHWP angle
    fig, ax = plt.subplots(2, 1, figsize=(10,5), sharex=True)
    ax[0].scatter(qp_ls, phi_ls, color='blue', label='reconst')
    # add sweep data
    qp_sweep = pd.read_csv(join(DATA_PATH, 'qp_phi_sweep_30.csv'))
    # qp_sweep['QP'] = 360 + qp_sweep['QP']
    qp_sweep['phi'] = 360 - qp_sweep['phi']
    ax[0].scatter(qp_sweep['QP'], qp_sweep['phi'], color='red', label='sweep')
    ax[1].scatter(qp_ls, fidelity_ls, color='blue', label='fidelity')
    ax[1].set_xlabel('QP angle $(\degree)$')
    ax[1].set_ylabel('Fidelity')
    ax[0].set_ylabel('$\phi (\degree)$')
    plt.suptitle('QP angle vs $\phi$')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, 'qp_phi_reconstr.pdf'))
    
def det_UVHWP_phi():
    phi_ls = []
    fidelity_ls = []
    uv_ls = np.linspace(2, 43, 6)
    # swept 2 to 43 in 6 steps
    for uv in uv_ls:
        file = f'rho_uv_{uv}.npy'
        fp = find_phi(file)
        phi = np.rad2deg(fp[0])
        if phi < 0:
            phi += 360
        phi_ls.append(phi)
        fidelity_ls.append(fp[1])

    # save csv
    uvhwp_phi = pd.DataFrame({'UVHWP': uv_ls, 'phi': phi_ls, 'fidelity': fidelity_ls})
    uvhwp_phi.to_csv(join(DATA_PATH, 'uvhwp_phi_reconstr.csv'), index=False)

    # uvhwp_phi = pd.read_csv(join(DATA_PATH, 'uvhwp_phi_reconstr.csv'))
    # uv_ls = uvhwp_phi['UVHWP']
    # phi_ls = uvhwp_phi['phi']
    # fidelity_ls = uvhwp_phi['fidelity']

    # plot phi vs UVHWP angle
    fig, ax = plt.subplots(2, 1, figsize=(10,5), sharex=True)
    ax[0].scatter(uv_ls, phi_ls, color='blue', label='reconst')
    # add sweep data
    uv_sweep = pd.read_csv(join(DATA_PATH, 'uvhwp_phi_sweep_30.csv'))
    ax[0].scatter(uv_sweep['UV_HWP'], uv_sweep['phi'], color='red', label='sweep')
    ax[1].scatter(uv_ls, fidelity_ls, color='blue', label='fidelity')
    ax[1].set_xlabel('UVHWP angle $(\degree)$')
    ax[1].set_ylabel('Fidelity')
    ax[0].set_ylabel('$\phi (\degree)$')
    plt.suptitle('UVHWP angle vs $\phi$')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, 'uvhwp_phi_reconstr.pdf'))

if __name__ == '__main__':
    det_UVHWP_phi()
    det_QP_phi()

    


    

    
    

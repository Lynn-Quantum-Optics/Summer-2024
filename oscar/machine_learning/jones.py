# file for sample jones matrix computations

# main package imports #
import numpy as np
from scipy.optimize import minimize

# special methods for density matrices #
from rho_methods import check_conc_min_eig, is_valid_rho

## jones matrices ##
def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): return R(alpha) @ np.matrix(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def get_QP(phi): return np.matrix(np.diag([1, np.e**(phi*1j)]))
B = np.matrix([[0, 0, 0, 1], [1, 0,0,0]]).T
init_state = np.matrix([[1,0],[0,0]])

def get_Jrho_C(angles):
    ''' Jones matrix with *almost* current setup, just adding one QWP on Alice. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_Detectors
    '''
    H1 = H(angles[0])
    H2 = H(angles[1])
    Q1 = Q(angles[2])
    Q2 = Q(angles[3])
    QP = get_QP(angles[4])

    ## compute density matrix ##
    P = np.kron(Q2, Q1 @ H2) @ B @ QP @ H1 @ init_state
    rho = np.round(P @ P.H,2)

    return rho

def get_Jrho_I(angles):
    ''' Jones matrix method with Ideal setup. Computes the rotated polarization state using the following setup:
        UV_HWP -> QP -> BBO -> A_HWP -> A_QWP -> A_QWP -> A_Detectors
                            -> B_HWP -> B_QWP -> B_QWP -> B_Detectors
    '''
    H1 = H(angles[0])
    H2 = H(angles[1])
    H3 = H(angles[2])
    Q1 = Q(angles[3])
    Q2 = Q(angles[4])
    Q3 = Q(angles[5])
    Q4 = Q(angles[6])
    Q5 = Q(angles[7])
    Q6 = Q(angles[8])

    ## compute density matrix ##
    P = np.kron(H3 @ Q5 @ Q6, H2 @ Q3 @ Q4) @ B @Q2 @Q1 @ H1 @ init_state
    rho = np.round(P @ P.H,2)

    return rho

def jones_decompose(func, targ_rho):
    ''' Function to decompose a given density matrix into jones matrices. 
    params:
        func: what setup (either Jrho_C or Jrho_I) to fit to
        targ_rho: target density matrix
    '''
    def get_random_angles_C():
        ''' Returns random angles for the Jrho_C setup'''
        theta_ls = np.random.rand(2)*np.pi/4
        theta1, theta2 = theta_ls[0], theta_ls[1]
        alpha_ls = np.random.rand(2)*np.pi/2
        alpha1, alpha2 = alpha_ls[0], alpha_ls[1]
        phi = np.random.rand()*0.69 # experimental limit of our QP

        return [theta1, theta2, alpha1, alpha2, phi]
    
    # initial guesses (PhiP)
    if func==get_Jrho_C:
        # x0 = [np.pi/8,0,0, 0, 0]
        x0 = get_random_angles_C()
        bounds = [(0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2), (0, 0.69)]
    elif func==get_Jrho_I:
        x0 = [np.pi/8,0,0, 0,0,0, 0,0,0]
        bounds = bounds = [(0, np.pi/4), (0, np.pi/4), (0, np.pi/4), (0, np.pi/2), (0, np.pi/2),(0, np.pi/2), (0, np.pi/2), (0, np.pi/2), (0, np.pi/2) ]

    # loss function
    def loss(angles, targ_rho):
        # get predicted density matrix
        pred_rho = func(angles)
        # get element-wise squared diff
        squared_diff = np.square(pred_rho - targ_rho)
        # get RMSE
        sum_squared_diff = np.sqrt(1 / (2*targ_rho.shape[0])**2 * np.sum(squared_diff))
        return sum_squared_diff

    # minimize loss function
    min_result= minimize(loss, x0=x0, args=(targ_rho), bounds=bounds)
    min_loss = min_result.fun
    best_angles = min_result.x
    print('actual state', targ_rho)
    print('predicted state', func(best_angles) )
    return best_angles, min_loss


if __name__=='__main__':
    # import predefined states for testing
    from sample_rho import PhiP, PhiM, PsiP, PsiM

    # angles=[np.pi/8,0,0, 0, 0] # PhiP
    # angles=[np.pi/8, np.pi/4, 0, 0, np.pi] # PsiM
    # rho = get_Jrho_C(angles)
    # print(rho)
    from roik_datagen import get_random_roik
    jones_decompose(get_Jrho_C, get_random_roik())
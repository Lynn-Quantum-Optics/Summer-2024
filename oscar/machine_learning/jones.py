# file for sample jones matrix computations
import numpy as np

# from main file
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

if __name__=='__main__':
    # import predefined states for testing
    from sample_rho import PhiP, PhiM, PsiP, PsiM

    angles=[np.pi/8,0,0, 0, 0] # PhiP
    rho = get_Jrho_C(*angles)
    print(rho)
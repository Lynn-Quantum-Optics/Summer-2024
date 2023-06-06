# file to use the jones matrices to randomly generate entangled states
import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import minimize
from tqdm import trange # for progress bar

## jones matrices ##
def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
def H(theta): return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
def Q(alpha): return R(alpha) @ np.matrix(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
def get_QP(phi): return np.matrix(np.diag([1, np.e**(phi*1j)]))

## Phi+ Bell state density matrix ##
PhiP = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
rho_PhiP= PhiP @ PhiP.reshape((1,4))

## function to compute permutation ##
def compute_state(theta1, theta2, alpha1, alpha2, phi):
    '''
    Computes the rotated polarization state given:
    --
    theta1: angle for UV_HWP (0 -> pi/4)
    theta2: angle for Bob C_HWP (0 -> pi/4)
    alpha1: angle for QWP (0 -> pi/2)
    phi1: angle for QP (0 -> 0.69 (due to limitations of experiment))
    '''
    H1 = H(theta1)
    H2 = H(theta2)
    Q1 = Q(alpha1)
    Q2 = Q(alpha2)
    QP = get_QP(phi)

    ## compute density matrix ##
    M = np.kron(H1, H2 @ Q2 @ Q1 @ QP @ H1) @ rho_PhiP
    rho = np.round(M @ M.H,2).real

    return rho

def get_random_state():
    '''
    Computes random angles in the ranges specified and generates the resulting states
    '''
    theta_ls = np.random.rand(2)*np.pi/4
    theta1, theta2 = theta_ls[0], theta_ls[1]
    alpha_ls = np.random.rand(2)*np.pi/2
    alpha1, alpha2 = alpha_ls[0], alpha_ls[1]
    phi = np.random.rand()*0.69

    return compute_state(theta1, theta2, alpha1, alpha2, phi), [theta1, theta2, alpha1, alpha2, phi]

def analyze_state(rho_angles):
    '''
    Takes as input a density matrix, outputs dictionary with angles, projection probabilities, W values, and W' values
    '''
    rho = rho_angles[0]
    angles = rho_angles[1]
    def get_expec_vals(rho):
        '''
        Returns all 16 expectation vals given density matrix.
        '''
        # pauli matrices
        I = np.eye(2)
        X = np.matrix([[0, 1], [1, 0]])
        Y = np.matrix([[0, -1j], [1j, 0]])
        Z = np.matrix([[1, 0], [0, -1]])

        expec_vals=[]

        Pa_matrices = [I, X, Y, Z]
        for Pa1 in Pa_matrices:
            for Pa2 in Pa_matrices:
                expec_vals.append(np.trace(np.kron(Pa1, Pa2) @ rho))

        return np.array(expec_vals).reshape(4,4)

    def compute_proj(basis1, basis2, rho):
        '''
        Computes projection into desired bases.
        '''
        return  np.trace(rho @ np.kron(basis1, basis2))

    def compute_witnesses(rho):
        '''
        Computes the 6 Ws and 9 W's.
        '''
        expec_vals = get_expec_vals(rho)

        def get_W1(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return (0.25*(expec_vals[0,0] + expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] + (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[3,0] + expec_vals[0,3]))).real
        def get_W2(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return (0.25*(expec_vals[0,0] - expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] - (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[3,0] - expec_vals[0,3]))).real
        def get_W3(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return (0.25*(expec_vals[0,0] + expec_vals[1,1] + (a**2 - b**2)*expec_vals[3,3] + (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[1,0] + expec_vals[0,1]))).real
        def get_W4(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return (0.25*(expec_vals[0,0] - expec_vals[1,1] + (a**2 - b**2)*expec_vals[3,3] - (a**2 - b**2)*expec_vals[2,2] - 2*a*b*(expec_vals[1,0] - expec_vals[0,1]))).real
        def get_W5(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return (0.25*(expec_vals[0,0] + expec_vals[2,2] + (a**2 - b**2)*expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] + 2*a*b*(expec_vals[2,0] + expec_vals[0,2]))).real
        def get_W6(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return (0.25*(expec_vals[0,0] - expec_vals[2,2] + (a**2 - b**2)*expec_vals[3,3] - (a**2 - b**2)*expec_vals[1,1] - 2*a*b*(expec_vals[2,0] - expec_vals[0,2]))).real
        
        ## W' from summer 2022 ##
        def get_Wp1(params, expec_vals):
            theta, alpha = params[0], params[1]
            return (.25*(expec_vals[0,0] + expec_vals[3,3] + np.cos(2*theta)*(expec_vals[1,1]+expec_vals[2,2])+np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.sin(alpha)*(expec_vals[1,2] - expec_vals[2,1]))).real
        def get_Wp2(params, expec_vals):
            theta, alpha = params[0], params[1]
            return (.25*(expec_vals[0,0] - expec_vals[3,3] + np.cos(2*theta)*(expec_vals[1,1]-expec_vals[2,2])+np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,0] - expec_vals[0,3]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[1,2] - expec_vals[2,1]))).real
        def get_Wp3(params, expec_vals):
            theta, alpha, beta = params[0], params[1], params[2]
            return (.25 * (np.cos(theta)**2*(expec_vals[0,0] + expec_vals[3,3]) + np.sin(theta)**2*(expec_vals[0,0] - expec_vals[3,3]) + np.cos(theta)**2*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(theta)**2*np.cos(2*alpha - beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*expec_vals[1,0] + np.sin(2*theta)*np.cos(alpha - beta)*expec_vals[0,1] + np.sin(2*theta)*np.sin(alpha)*expec_vals[2,0] + np.sin(2*theta)*np.sin(alpha - beta)*expec_vals[0,2]+np.cos(theta)**2*np.sin(beta)*(expec_vals[2,1] - expec_vals[1,2]) + np.sin(theta)**2*np.sin(2*alpha - beta)*(expec_vals[2,1] + expec_vals[1,2]))).real
        def get_Wp4(params, expec_vals):
            theta, alpha = params[0], params[1]
            return (.25*(expec_vals[0,0]+expec_vals[1,1]+np.cos(2*theta)*(expec_vals[3,3] + expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[0,1] + expec_vals[1,0]) + np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,3] - expec_vals[3,2]))).real
        def get_Wp5(params, expec_vals):
            theta, alpha = params[0], params[1]
            return (.25*(expec_vals[0,0]-expec_vals[1,1]+np.cos(2*theta)*(expec_vals[3,3] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[0,1] - expec_vals[1,0]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,3] - expec_vals[3,2]))).real
        def get_Wp6(params,expec_vals):
            theta, alpha, beta = params[0], params[1], params[2]
            return (.25*(np.cos(theta)**2*np.cos(alpha)**2*(expec_vals[0,0] + expec_vals[3,3] + expec_vals[3,0] + expec_vals[0,3]) + np.cos(theta)**2*np.sin(alpha)**2*(expec_vals[0,0] - expec_vals[3,3] + expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.cos(beta)**2*(expec_vals[0,0] + expec_vals[3,3] - expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.sin(beta)**2*(expec_vals[0,0] - expec_vals[3,3] - expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(expec_vals[2,3] + expec_vals[2,0]) + np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(expec_vals[2,3] - expec_vals[2,0]) - np.cos(theta)**2*np.sin(2*alpha)*(expec_vals[3,2] + expec_vals[0,2]) - np.sin(theta)**2*np.sin(2*beta)*(expec_vals[3,2] - expec_vals[0,2]))).real
        def get_Wp7(params, expec_vals):
            theta, alpha = params[0], params[1]
            return (.25*(expec_vals[0,0] + expec_vals[2,2]+np.cos(2*theta)*(expec_vals[3,3] + expec_vals[1,1]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,1] - expec_vals[1,3]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,0]+expec_vals[0,2]))).real
        def get_Wp8(params, expec_vals):
            theta, alpha = params[0], params[1]
            return (.25*(expec_vals[0,0] - expec_vals[2,2] + np.cos(2*theta)*(expec_vals[3,3]-expec_vals[1,1]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,1]+expec_vals[1,3])+np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,0] - expec_vals[0,2]))).real
        def get_Wp9(params, expec_vals):
            theta, alpha, beta = params[0], params[1], params[2]
            return (.25*(np.cos(theta)**2*np.cos(alpha)**2*(expec_vals[0,0] + expec_vals[3,3] + expec_vals[3,0] + expec_vals[0,3]) + np.cos(theta)**2*np.sin(alpha)**2*(expec_vals[0,0] - expec_vals[3,3] + expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.cos(beta)**2*(expec_vals[0,0] + expec_vals[3,3] - expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.sin(beta)**2*(expec_vals[0,0] - expec_vals[3,3] - expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.cos(theta)**2*np.sin(2*alpha)*(expec_vals[0,1] + expec_vals[3,1]) + np.sin(theta)**2*np.sin(2*beta)*(expec_vals[0,1] - expec_vals[3,1]) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(expec_vals[1,0] + expec_vals[1,3])+ np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(expec_vals[1,0] - expec_vals[1,3]))).real

        # print('rho', rho)
        # print('expec vals', expec_vals)
        # print('testing Wp1', get_Wp1([np.pi, np.pi/3], expec_vals))

        # now perform optimization; break into three groups based on the number of params to optimize
        all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
        W_expec_vals = []
        for i, W in enumerate(all_W):
            if i <= 5: # just theta optimization
                W_expec_vals.append(minimize(W, x0=[0], args = (expec_vals,), bounds=[(0, np.pi/2)])['fun'][0])
            elif i==8 or i==11 or i==14: # theta, alpha, and beta
                 W_expec_vals.append(minimize(W, x0=[0, 0, 0], args = (expec_vals,), bounds=[(0, np.pi/2), (0, 2*np.pi), (0, 2*np.pi)])['fun'])
            else:# theta and alpha
                W_expec_vals.append(minimize(W, x0=[0, 0], args = (expec_vals,), bounds=[(0, np.pi/2), (0, 2*np.pi)])['fun'])
        
        # find min W expec value; this tells us if first 12 measurements are enough #
        W_min = min(W_expec_vals[:6])
        Wp_t1 = min(W_expec_vals[6:9])
        Wp_t2 = min(W_expec_vals[9:12])
        Wp_t3 = min(W_expec_vals[12:15])

        return W_min, Wp_t1, Wp_t2, Wp_t3


    ## define the single qubit bases for projection: still want our input vectors to be these probabilities ##
    H = np.array([[1,0]]).reshape(2,1) # ZP
    V = np.array([[0,1]]).reshape(2,1) # ZM
    D = 1/np.sqrt(2) * np.array([[1,1]]).reshape(2,1) # XP
    A = 1/np.sqrt(2) * np.array([[1,-1]]).reshape(2,1) # XM
    R = 1/np.sqrt(2) * np.array([[1,1j]]).reshape(2,1) # YP
    L = 1/np.sqrt(2) * np.array([[1,-1j]]).reshape(2,1) # YM

    HH = compute_proj(H, H, rho)
    HV = compute_proj(H, V, rho)
    VH = compute_proj(V, H, rho)
    VV = compute_proj(V, V, rho)
    DD = compute_proj(D, D, rho)
    DA = compute_proj(D, A, rho)
    AD = compute_proj(A, D, rho)
    AA = compute_proj(A, A, rho)
    RR = compute_proj(R, R, rho)
    RL = compute_proj(R, L, rho)
    LR = compute_proj(L, R, rho)
    LL = compute_proj(L, L, rho)

    ## compute W and W' ##
    W_min, Wp_t1, Wp_t2, Wp_t3 = compute_witnesses(rho)
   

    return {'theta1':angles[0], 'theta2':angles[1], 'alpha1':angles[2], 'alpha2':angles[3], 'phi':angles[4],
     'HH':HH, 'HV':HV,'VH':VH, 'VV':VV, 'DD':DD, 'DA':DA, 'AD':AD, 'AA':AA, 
     'RR':RR, 'RL':RL, 'LR':LR, 'LL':LL, 'W_min':W_min, 'Wp_t1': Wp_t1,'Wp_t2': Wp_t2, 'Wp_t3': Wp_t3}

## perform randomization ##
# set number of states
N=102000 
# initilize dataframe to hold states
df = pd.DataFrame({'theta1':[], 'theta2':[], 'alpha1':[], 'alpha2':[], 'phi':[],
     'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
     'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': []}) 
for i in trange(N):
    analyze_state(get_random_state())
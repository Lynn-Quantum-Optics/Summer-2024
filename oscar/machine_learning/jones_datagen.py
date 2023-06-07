# file to use the jones matrices to randomly generate entangled states
from os.path import join
import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import minimize
import scipy.linalg as la
from tqdm import trange # for progress bar

def get_random_jones():
    '''
    Computes random angles in the ranges specified and generates the resulting states
    '''
    ## jones matrices ##
    def R(alpha): return np.matrix([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    def H(theta): return np.matrix([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])
    def Q(alpha): return R(alpha) @ np.matrix(np.diag([np.e**(np.pi / 4 * 1j), np.e**(-np.pi / 4 * 1j)])) @ R(-alpha)
    def get_QP(phi): return np.matrix(np.diag([1, np.e**(phi*1j)]))
    B = np.matrix([[0, 0, 0, 1], [1, 0,0,0]]).T

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
        P = np.kron(Q2,Q1 @ H2) @ B @ QP @ H1
        rho = np.real(np.round(P @ P.H,2))

        return np.matrix(rho)

    theta_ls = np.random.rand(2)*np.pi/4
    theta1, theta2 = theta_ls[0], theta_ls[1]
    alpha_ls = np.random.rand(2)*np.pi/2
    alpha1, alpha2 = alpha_ls[0], alpha_ls[1]
    phi = np.random.rand()*0.69

    return [compute_state(theta1, theta2, alpha1, alpha2, phi), [theta1, theta2, alpha1, alpha2, phi]]

def get_random_simplex():
    '''
    Returns density matrix for random state of form:
    a|HH> + be^(i*beta)|01> + ce^(i*gamma)*|10> + de^(i*delta)*|11>
    '''
    
    a = np.random.rand()
    b = np.random.rand() *(1-a)
    c = np.random.rand()*(1-a-b)
    d = 1-a-b-c

    real_ls = [] # list to store real coefficients
    real_ls = np.sqrt(np.array([a, b, c, d]))
    np.random.shuffle(real_ls)
    rand_angle=np.random.rand(3)*2*np.pi

    state_vec = np.multiply(real_ls, np.e**(np.concatenate((np.array([1]), rand_angle))*1j)).reshape((4,1))

    # compute density matrix
    rho = np.matrix(np.real(state_vec @ np.conjugate(state_vec.reshape((1,4)))))

    return [rho, np.concatenate((real_ls, rand_angle))]

def analyze_state(rho_angles, rand_type):
    '''
    Takes as input a density matrix, outputs dictionary with angles, projection probabilities, W values, and W' values. 
    rand_type is a string, either <jones> or <simplex>, which builds the correct df
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
        return  np.real(np.trace(rho @ np.kron(basis1, basis2)))

    def compute_witnesses(rho):
        '''
        Computes the 6 Ws and 9 W's.
        '''
        expec_vals = get_expec_vals(rho)

        def get_W1(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return np.real(0.25*(expec_vals[0,0] + expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] + (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[3,0] + expec_vals[0,3])))
        def get_W2(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return np.real(0.25*(expec_vals[0,0] - expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] - (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[3,0] - expec_vals[0,3])))
        def get_W3(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return np.real(0.25*(expec_vals[0,0] + expec_vals[1,1] + (a**2 - b**2)*expec_vals[3,3] + (a**2 - b**2)*expec_vals[2,2] + 2*a*b*(expec_vals[1,0] + expec_vals[0,1])))
        def get_W4(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return np.real(0.25*(expec_vals[0,0] - expec_vals[1,1] + (a**2 - b**2)*expec_vals[3,3] - (a**2 - b**2)*expec_vals[2,2] - 2*a*b*(expec_vals[1,0] - expec_vals[0,1])))
        def get_W5(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return np.real(0.25*(expec_vals[0,0] + expec_vals[2,2] + (a**2 - b**2)*expec_vals[3,3] + (a**2 - b**2)*expec_vals[1,1] + 2*a*b*(expec_vals[2,0] + expec_vals[0,2])))
        def get_W6(theta, expec_vals):
            a, b = np.cos(theta), np.sin(theta)
            return np.real(0.25*(expec_vals[0,0] - expec_vals[2,2] + (a**2 - b**2)*expec_vals[3,3] - (a**2 - b**2)*expec_vals[1,1] - 2*a*b*(expec_vals[2,0] - expec_vals[0,2])))
        
        ## W' from summer 2022 ##
        def get_Wp1(params, expec_vals):
            theta, alpha = params[0], params[1]
            return np.real(.25*(expec_vals[0,0] + expec_vals[3,3] + np.cos(2*theta)*(expec_vals[1,1]+expec_vals[2,2])+np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.sin(alpha)*(expec_vals[1,2] - expec_vals[2,1])))
        def get_Wp2(params, expec_vals):
            theta, alpha = params[0], params[1]
            return np.real(.25*(expec_vals[0,0] - expec_vals[3,3] + np.cos(2*theta)*(expec_vals[1,1]-expec_vals[2,2])+np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,0] - expec_vals[0,3]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[1,2] - expec_vals[2,1])))
        def get_Wp3(params, expec_vals):
            theta, alpha, beta = params[0], params[1], params[2]
            return np.real(.25 * (np.cos(theta)**2*(expec_vals[0,0] + expec_vals[3,3]) + np.sin(theta)**2*(expec_vals[0,0] - expec_vals[3,3]) + np.cos(theta)**2*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(theta)**2*np.cos(2*alpha - beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*expec_vals[1,0] + np.sin(2*theta)*np.cos(alpha - beta)*expec_vals[0,1] + np.sin(2*theta)*np.sin(alpha)*expec_vals[2,0] + np.sin(2*theta)*np.sin(alpha - beta)*expec_vals[0,2]+np.cos(theta)**2*np.sin(beta)*(expec_vals[2,1] - expec_vals[1,2]) + np.sin(theta)**2*np.sin(2*alpha - beta)*(expec_vals[2,1] + expec_vals[1,2])))
        def get_Wp4(params, expec_vals):
            theta, alpha = params[0], params[1]
            return np.real(.25*(expec_vals[0,0]+expec_vals[1,1]+np.cos(2*theta)*(expec_vals[3,3] + expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[0,1] + expec_vals[1,0]) + np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,3] - expec_vals[3,2])))
        def get_Wp5(params, expec_vals):
            theta, alpha = params[0], params[1]
            return np.real(.25*(expec_vals[0,0]-expec_vals[1,1]+np.cos(2*theta)*(expec_vals[3,3] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[0,1] - expec_vals[1,0]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,3] - expec_vals[3,2])))
        def get_Wp6(params,expec_vals):
            theta, alpha, beta = params[0], params[1], params[2]
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(expec_vals[0,0] + expec_vals[3,3] + expec_vals[3,0] + expec_vals[0,3]) + np.cos(theta)**2*np.sin(alpha)**2*(expec_vals[0,0] - expec_vals[3,3] + expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.cos(beta)**2*(expec_vals[0,0] + expec_vals[3,3] - expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.sin(beta)**2*(expec_vals[0,0] - expec_vals[3,3] - expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(expec_vals[2,3] + expec_vals[2,0]) + np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(expec_vals[2,3] - expec_vals[2,0]) - np.cos(theta)**2*np.sin(2*alpha)*(expec_vals[3,2] + expec_vals[0,2]) - np.sin(theta)**2*np.sin(2*beta)*(expec_vals[3,2] - expec_vals[0,2])))
        def get_Wp7(params, expec_vals):
            theta, alpha = params[0], params[1]
            return np.real(.25*(expec_vals[0,0] + expec_vals[2,2]+np.cos(2*theta)*(expec_vals[3,3] + expec_vals[1,1]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,1] - expec_vals[1,3]) - np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,0]+expec_vals[0,2])))
        def get_Wp8(params, expec_vals):
            theta, alpha = params[0], params[1]
            return np.real(.25*(expec_vals[0,0] - expec_vals[2,2] + np.cos(2*theta)*(expec_vals[3,3]-expec_vals[1,1]) + np.sin(2*theta)*np.cos(alpha)*(expec_vals[3,1]+expec_vals[1,3])+np.sin(2*theta)*np.sin(alpha)*(expec_vals[2,0] - expec_vals[0,2])))
        def get_Wp9(params, expec_vals):
            theta, alpha, beta = params[0], params[1], params[2]
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(expec_vals[0,0] + expec_vals[3,3] + expec_vals[3,0] + expec_vals[0,3]) + np.cos(theta)**2*np.sin(alpha)**2*(expec_vals[0,0] - expec_vals[3,3] + expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.cos(beta)**2*(expec_vals[0,0] + expec_vals[3,3] - expec_vals[3,0] - expec_vals[0,3]) + np.sin(theta)**2*np.sin(beta)**2*(expec_vals[0,0] - expec_vals[3,3] - expec_vals[3,0] + expec_vals[0,3]) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(expec_vals[1,1] + expec_vals[2,2]) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(expec_vals[1,1] - expec_vals[2,2]) + np.cos(theta)**2*np.sin(2*alpha)*(expec_vals[0,1] + expec_vals[3,1]) + np.sin(theta)**2*np.sin(2*beta)*(expec_vals[0,1] - expec_vals[3,1]) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(expec_vals[1,0] + expec_vals[1,3])+ np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(expec_vals[1,0] - expec_vals[1,3])))

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
        W_min = np.real(min(W_expec_vals[:6]))
        Wp_t1 = np.real(min(W_expec_vals[6:9]))
        Wp_t2 = np.real(min(W_expec_vals[9:12]))
        Wp_t3 = np.real(min(W_expec_vals[12:15]))

        return W_min, Wp_t1, Wp_t2, Wp_t3
    
    def get_min_eig(M0):
        '''
        Computes the eigenvalues of the partial transpose; if at least one is negative, then state labeled as '0' for entangled; else, '1'. 
        '''
        def partial_transpose(M0):
            # decompose M0 into blocks
            b1 = M0[:2, :2]
            b2 = M0[:2, 2:]
            b3 = M0[2:, :2]
            b4 = M0[2:, 2:]

            PT = np.matrix(np.block([[b1.T, b2.T], [b3.T, b4.T]]))
            return PT

        # compute partial tranpose
        PT = partial_transpose(M0)
        eigenvals = la.eigvals(PT)
        eigenvals.sort() # sort

        return np.real(eigenvals[0]) # return min eigenvalue

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

    min_eig = get_min_eig(rho)
   
    if rand_type=='jones':
        return {'theta1':angles[0], 'theta2':angles[1], 'alpha1':angles[2], 'alpha2':angles[3], 'phi':angles[4],
        'HH':HH, 'HV':HV,'VH':VH, 'VV':VV, 'DD':DD, 'DA':DA, 'AD':AD, 'AA':AA, 
        'RR':RR, 'RL':RL, 'LR':LR, 'LL':LL, 'W_min':W_min, 'Wp_t1': Wp_t1,'Wp_t2': Wp_t2, 'Wp_t3': Wp_t3, 'min_eig':min_eig}
    elif rand_type =='simplex':
        return {'a':angles[0], 'b':angles[1], 'c':angles[2], 'd':angles[3], 'beta':angles[4], 'gamma':angles[5], 'delta':angles[6],
        'HH':HH, 'HV':HV,'VH':VH, 'VV':VV, 'DD':DD, 'DA':DA, 'AD':AD, 'AA':AA, 
        'RR':RR, 'RL':RL, 'LR':LR, 'LL':LL, 'W_min':W_min, 'Wp_t1': Wp_t1,'Wp_t2': Wp_t2, 'Wp_t3': Wp_t3, 'min_eig':min_eig}
    else:
        print('Incorrect rand_type.')

## perform randomization ##
def gen_data(N=50000):
    # initilize dataframe to hold states
    df_jones = pd.DataFrame({'theta1':[], 'theta2':[], 'alpha1':[], 'alpha2':[], 'phi':[],
        'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
        'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
    df_simplex = pd.DataFrame({'a':[], 'b':[], 'c':[], 'd':[], 'beta':[], 'gamma':[], 'delta':[],
        'HH':[], 'HV':[],'VH':[], 'VV':[], 'DD':[], 'DA':[], 'AD':[], 'AA':[], 
        'RR':[], 'RL':[], 'LR':[], 'LL':[], 'W_min':[], 'Wp_t1': [],'Wp_t2': [], 'Wp_t3': [], 'min_eig':[]}) 
    for i in trange(N):
        df_jones = df_jones.append(analyze_state(get_random_jones(), 'jones'), ignore_index=True)
        df_simplex = df_simplex.append(analyze_state(get_random_simplex(), 'simplex'), ignore_index=True)

    # save!
    DATA_PATH = 'jones_simplex_data'
    df_jones.to_csv(join(DATA_PATH, 'jones_%i_0.csv'%N))
    df_simplex.to_csv(join(DATA_PATH, 'simplex_%i_0.csv'%N))
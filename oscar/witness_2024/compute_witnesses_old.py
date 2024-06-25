import numpy as np
from scipy.optimize import minimize, approx_fprime
from uncertainties import unumpy as unp

def adjoint(state):
    ''' Returns the adjoint of a state vector. For a np.matrix, can use .H'''
    return np.conjugate(state).T

def partial_transpose(rho, subsys='B'):
    ''' Helper function to compute the partial transpose of a density matrix. Useful for the Peres-Horodecki criterion, which states that if the partial transpose of a density matrix has at least one negative eigenvalue, then the state is entangled.
    Params:
        rho: density matrix
        subsys: which subsystem to compute partial transpose wrt, i.e. 'A' or 'B'
    '''
    # decompose rho into blocks
    b1 = rho[:2, :2]
    b2 = rho[:2, 2:]
    b3 = rho[2:, :2]
    b4 = rho[2:, 2:]

    PT = np.matrix(np.block([[b1.T, b2.T], [b3.T, b4.T]]))

    if subsys=='B':
        return PT
    elif subsys=='A':
        return PT.T

def compute_witnesses_old(rho, counts = None, expt = False, do_counts = False, expt_purity = None, model=None, do_W = False, do_richard = False, UV_HWP_offset=None, angles = None, num_reps = 30, optimize = True, gd=True, zeta=0.7, ads_test=False, return_all=False, return_params=False, return_lynn=False, return_lynn_only=False):
    ''' Computes the minimum of the 6 Ws and the minimum of the 3 triples of the 9 W's. 
        Params:
            rho: the density matrix
            counts: raw unp array of counts and unc
            expt: bool, whether to compute the Ws assuming input is experimental data
            do_stokes: bool, whether to compute 
            do_counts: use the raw definition in terms of counts
            expt_purity: the experimental purity of the state, which defines the noise level: 1 - purity.
            model: which model to correct for noise; see det_noise in process_expt.py for more info
            do_W: bool, whether to use W calc in loss for noise
            UV_HWP_offset: see description in det_noise in process_expt.py
            model_path: path to noise model csvs.
            angles: angles of eta, chi for E0 states to adjust theory
            num_reps: int, number of times to run the optimization
            optimize: bool, whether to optimize the Ws with random or gradient descent or to just check bounds
            gd: bool, whether to use gradient descent or brute random search
            zeta: learning rate for gradient descent
            ads_test: bool, whether to return w2 expec and sin (theta) for the amplitude damped states
            return_all: bool, whether to return all the Ws or just the min of the 6 and the min of the 3 triples
            return_params: bool, whether to return the params that give the min of the 6 and the min of the 3 triples
    '''

    # check if experimental data
    if expt and counts is not None:
        do_counts = True

    # if wanting to account for experimental purity, add noise to the density matrix for adjusted theoretical purity calculation

    # automatic correction is depricated; send the theoretical rho after whatever correction you want to this function

        # if expt_purity is not None and angles is not None: # this rho is theoretical
        #     if model is None:
        #         rho = adjust_rho(rho, angles, expt_purity)
        #     else:
        #         rho = load_saved_get_E0_rho_c(rho, angles, expt_purity, model, UV_HWP_offset, do_W = do_W, do_richard = do_richard)
        #     # rho = adjust_rho(rho, angles, expt_purity)

    if do_counts:
        counts = np.reshape(counts, (36,1))
        def get_W1(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W2(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W3(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W4(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) - 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W5(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_W6(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        
        ## W' from summer 2022 ##
        def get_Wp1(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))+((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) - ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp2(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))-((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) - np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) - ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp3(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25 * (np.cos(theta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(theta)**2*np.cos(2*alpha - beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*((DD + DA - AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.cos(alpha - beta)*((DD - DA + AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.sin(alpha)*((RR - LR + RL - LL) / (RR + LR + RL + LL)) + np.sin(2*theta)*np.sin(alpha - beta)*((RR + LR - RL - LL) / (RR + LR + RL + LL))+np.cos(theta)**2*np.sin(beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) - ((DR - DL - AR + AL) / (DR + DL + AR + AL))) + np.sin(theta)**2*np.sin(2*alpha - beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) + ((DR - DL - AR + AL) / (DR + DL + AR + AL)))))
        def get_Wp4(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1+((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((DD + DA - AD - AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp5(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1-((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((DD + DA - AD - AA) / (DD + DA + AD + AA))) - np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp6(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) + ((RR - LR + RL - LL) / (RR + LR + RL + LL))) + np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((RR - LR + RL - LL) / (RR + LR + RL + LL))) - np.cos(theta)**2*np.sin(2*alpha)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL))) - np.sin(theta)**2*np.sin(2*beta)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp7(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV))) - np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL))+((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp8(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV))-((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA))+((DH - DV - AH + AV) / (DH + DV + AH + AV)))+np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp9(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.cos(theta)**2*np.sin(2*alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(theta)**2*np.sin(2*beta)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DH - DV - AH + AV) / (DH + DV + AH + AV)))+ np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV)))))

        def get_nom(params, expec_vals, func):
            '''For use in error propagation; returns the nominal value of the function'''
            w = func(params, expec_vals)
            return unp.nominal_values(w)

        # now perform optimization; break into three groups based on the number of params to optimize
        all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
        W_expec_vals = []
        for i, W in enumerate(all_W):
            if i <= 5: # just theta optimization
                # get initial guess at boundary
                if not(expt):
                    def min_W(x0):
                        return minimize(W, x0=x0, args=(counts,), bounds=[(0, np.pi)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(counts, W), bounds=[(0, np.pi/2)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x

                x0 = [0]
                w0_val = min_W_val(x0)
                w0_params = min_W_params(x0)
                x0 = [np.pi]
                w1_val = min_W_val(x0)
                w1_params = min_W_params(x0)
                if w0_val < w1_val:
                    w_min_val = w0_val
                    w_min_params = w0_params
                else:
                    w_min_val = w1_val
                    w_min_params = w1_params
                if optimize:
                    isi = 0 # index since last improvement
                    for _ in range(num_reps): # repeat 10 times and take the minimum
                        if gd:
                            if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                x0 = [np.random.rand()*np.pi]
                            else:
                                grad = approx_fprime(x0, min_W_val, 1e-6)
                                if np.all(grad < 1e-5*np.ones(len(grad))):
                                    break
                                else:
                                    x0 = x0 - zeta*grad
                        else:
                            x0 = [np.random.rand()*np.pi]

                        w_val = min_W_val(x0)
                        w_params = min_W_params(x0)
                        
                        if w_val < w_min_val:
                            w_min_val = w_val
                            w_min_params = w_params
                            isi=0
                        else:
                            isi+=1
            elif i==8 or i==11 or i==14: # theta, alpha, and beta
                if not(expt):
                    def min_W(x0):
                        return minimize(W, x0=x0, args=(counts,), bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(counts, W), bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x
                    
                x0 = [0, 0, 0]
                w0_val = min_W_val(x0)
                w0_params = min_W_params(x0)
                x0 = [np.pi/2, 2*np.pi, 2*np.pi]
                w1_val = min_W_val(x0)
                w1_params = min_W_params(x0)
                if w0_val < w1_val:
                    w_min_val = w0_val
                    w_min_params = w0_params
                else:
                    w_min_val = w1_val
                    w_min_params = w1_params
                if optimize:
                    isi = 0 # index since last improvement
                    for _ in range(num_reps): # repeat 10 times and take the minimum
                        if gd:
                            if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                            else:
                                grad = approx_fprime(x0, min_W_val, 1e-6)
                                if np.all(grad < 1e-5*np.ones(len(grad))):
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                else:
                                    x0 = x0 - zeta*grad
                        else:
                            x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                        w_val = min_W_val(x0)
                        w_params = min_W_params(x0)
                        # print(w_min_val, w_val)
                        if w_val < w_min_val:
                            w_min_val = w_val
                            w_min_params = w_params
                            isi=0
                        else:
                            isi+=1
                
            else:# theta and alpha
                if not(expt):
                    def min_W(x0):
                        return minimize(W, x0=x0, args=(counts,), bounds=[(0, np.pi/2),(0, np.pi*2)])
                else:
                    def min_W(x0):
                        return minimize(get_nom, x0=x0, args=(counts, W), bounds=[(0, np.pi/2),(0, np.pi*2)])

                def min_W_val(x0):
                    return min_W(x0).fun

                def min_W_params(x0):
                    return min_W(x0).x
                    
                x0 = [0, 0]
                w0_val = min_W(x0).fun
                w0_params = min_W(x0).x
                x0 = [np.pi/2 , 2*np.pi]
                w1_val = min_W(x0).fun
                w1_params = min_W(x0).x
                if w0_val < w1_val:
                    w_min_val = w0_val
                    w_min_params = w0_params
                else:
                    w_min_val = w1_val
                    w_min_params = w1_params
                if optimize:
                    isi = 0 # index since last improvement
                    for _ in range(num_reps): # repeat 10 times and take the minimum
                        if gd:
                            if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                            else:
                                grad = approx_fprime(x0, min_W_val, 1e-6)
                                if np.all(grad < 1e-5*np.ones(len(grad))):
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                else:
                                    x0 = x0 - zeta*grad
                        else:
                            x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                        w_val = min_W_val(x0)
                        w_params = min_W_params(x0)
                        
                        if w_val < w_min_val:
                            w_min_val = w_val
                            w_min_params = w_params
                            isi=0
                        else:
                            isi+=1

            if expt: # automatically calculate uncertainty
                W_expec_vals.append(W(w_min_params, counts))
            else:
                W_expec_vals.append(w_min_val)
        W_min = np.real(min(W_expec_vals[:6]))
        try:
            Wp_t1 = np.real(min(W_expec_vals[6:9])[0])
            Wp_t2 = np.real(min(W_expec_vals[9:12])[0])
            Wp_t3 = np.real(min(W_expec_vals[12:15])[0])
        except TypeError:
            Wp_t1 = np.real(min(W_expec_vals[6:9]))
            Wp_t2 = np.real(min(W_expec_vals[9:12]))
            Wp_t3 = np.real(min(W_expec_vals[12:15]))
        
        return W_min, Wp_t1, Wp_t2, Wp_t3
        # return W_expec_vals

        
    else: # use operators instead like in eritas's matlab code
        # bell states #
        PHI_P = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
        PHI_M = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).reshape((4,1))
        PSI_P = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0]).reshape((4,1))
        PSI_M = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0]).reshape((4,1))
        # column vectors
        HH = np.array([1, 0, 0, 0]).reshape((4,1))
        HV = np.array([0, 1, 0, 0]).reshape((4,1))
        VH = np.array([0, 0, 1, 0]).reshape((4,1))
        VV = np.array([0, 0, 0, 1]).reshape((4,1))

        # get the operators
        # take rank 1 projector and return witness
        def get_witness(phi):
            ''' Helper function to compute the witness operator for a given state and return trace(W*rho) for a given state rho.'''
            W = phi * adjoint(phi)
            W = partial_transpose(W) # take partial transpose
            return np.real(np.trace(W @ rho))

        ## ------ for W ------ ##
        def get_W1(param):
            a,b = np.cos(param), np.sin(param)
            phi1 = a*PHI_P + b*PHI_M
            return get_witness(phi1)
        def get_W2(param):
            a,b = np.cos(param), np.sin(param)
            phi2 = a*PSI_P + b*PSI_M
            return get_witness(phi2)
        def get_W3(param):
            a,b = np.cos(param), np.sin(param)
            phi3 = a*PHI_P + b*PSI_P
            return get_witness(phi3)
        def get_W4(param):
            a,b = np.cos(param), np.sin(param)
            phi4 = a*PHI_M + b*PSI_M
            return get_witness(phi4)
        def get_W5(param):
            a,b = np.cos(param), np.sin(param)
            phi5 = a*PHI_P + 1j*b*PSI_M
            return get_witness(phi5)
        def get_W6(param):
            a,b = np.cos(param), np.sin(param)
            phi6 = a*PHI_M + 1j*b*PSI_P
            return get_witness(phi6)

        ## ------ for W' ------ ##
        def get_Wp1(params):
            theta, alpha = params[0], params[1]
            phi1_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PHI_M
            return get_witness(phi1_p)
        def get_Wp2(params):
            theta, alpha = params[0], params[1]
            phi2_p = np.cos(theta)*PSI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi2_p)
        def get_Wp3(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi3_p = 1/np.sqrt(2) * (np.cos(theta)*HH + np.exp(1j*(beta - alpha))*np.sin(theta)*HV + np.exp(1j*alpha)*np.sin(theta)*VH + np.exp(1j*beta)*np.cos(theta)*VV)
            return get_witness(phi3_p)
        def get_Wp4(params):
            theta, alpha = params[0], params[1]
            phi4_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_P
            return get_witness(phi4_p)
        def get_Wp5(params):
            theta, alpha = params[0], params[1]
            phi5_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi5_p)
        def get_Wp6(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi6_p = np.cos(theta)*np.cos(alpha)*HH + 1j * np.cos(theta)*np.sin(alpha)*HV + 1j * np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV
            return get_witness(phi6_p)
        def get_Wp7(params):
            theta, alpha = params[0], params[1]
            phi7_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi7_p)
        def get_Wp8(params):
            theta, alpha = params[0], params[1]
            phi8_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_P
            return get_witness(phi8_p)
        def get_Wp9(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi9_p = np.cos(theta)*np.cos(alpha)*HH + np.cos(theta)*np.sin(alpha)*HV + np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV
            return get_witness(phi9_p)
        
        def get_lynn():
            return 1/5*(2*HH +2*np.exp(1j*np.pi/4)*  HV +  np.exp(1j*np.pi/4)*VH +4*VV) 
        if return_lynn_only:
            return get_witness(get_lynn())
        # get the witness values by minimizing the witness function
        if not(ads_test): 
            all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
            W_expec_vals = []
            if return_params: # to log the params
                min_params = []
            for i, W in enumerate(all_W):
                if i <= 5: # just theta optimization
                    # get initial guess at boundary
                    def min_W(x0):
                        do_min = minimize(W, x0=x0, bounds=[(0, np.pi)])
                        # print(do_min['x'])
                        return do_min['fun']
                    x0 = [0]
                    w0 = min_W(x0)
                    x0 = [np.pi]
                    w1 = min_W(x0)
                    if w0 < w1:
                        w_min = w0
                        x0_best = [0]
                    else:
                        w_min = w1
                        x0_best = [np.pi]
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        break
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                    # print('------------------')
                elif i==8 or i==11 or i==14: # theta, alpha, and beta
                    def min_W(x0):
                        do_min = minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])
                        return do_min['fun']

                    x0 = [0, 0, 0]
                    w0 = min_W(x0)
                    x0 = [np.pi/2 , 2*np.pi, 2*np.pi]
                    w1 = min_W(x0)
                    if w0 < w1:
                        w_min = w0
                        x0_best = [0, 0, 0]
                    else:
                        w_min = w1
                        x0_best = [np.pi/2 , 2*np.pi, 2*np.pi]
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                else:# theta and alpha
                    def min_W(x0):
                        return minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2)])['fun']
                        
                    x0 = [0, 0]
                    w0 = min_W(x0)
                    x0 = [np.pi/2 , 2*np.pi]
                    w1 = min_W(x0)
                    if w0 < w1:
                        w_min = w0
                        x0_best = [0, 0]
                    else:
                        w_min = w1
                        x0_best = [np.pi/2 , 2*np.pi]
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                if return_params:
                    min_params.append(x0_best)
                W_expec_vals.append(w_min)
            # print('W', np.round(W_expec_vals[:6], 3))
            # print('W\'', np.round(W_expec_vals[6:], 3))
            # find min witness expectation values
            W_min = min(W_expec_vals[:6])
            Wp_t1 = min(W_expec_vals[6:9])
            Wp_t2 = min(W_expec_vals[9:12])
            Wp_t3 = min(W_expec_vals[12:15])
            # get the corresponding parameters
            if return_params:
                # sort by witness value; want the most negative, so take first element in sorted
                W_param = [x for _,x in sorted(zip(W_expec_vals[:6], min_params[:6]))][0]
                Wp_t1_param = [x for _,x in sorted(zip(W_expec_vals[6:9], min_params[6:9]))][0]
                Wp_t2_param = [x for _,x in sorted(zip(W_expec_vals[9:12], min_params[9:12]))][0]
                Wp_t3_param = [x for _,x in sorted(zip(W_expec_vals[12:15], min_params[12:15]))][0]

            # calculate lynn
            W_lynn = get_witness(get_lynn())

            if not(return_all):
                if return_params:
                    return W_min, Wp_t1, Wp_t2, Wp_t3, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                else:
                    if return_lynn:
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_lynn
                    else:
                        return W_min, Wp_t1, Wp_t2, Wp_t3
            else:
                if return_params:
                    return W_expec_vals, min_params
                else:
                    return W_expec_vals
        else: 
            W2_main= minimize(get_W2, x0=[0], bounds=[(0, np.pi)])
            W2_val = W2_main['fun']
            W2_param = W2_main['x']

            return W2_val, W2_param[0]

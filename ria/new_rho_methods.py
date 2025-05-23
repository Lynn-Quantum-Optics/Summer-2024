import numpy as np
from scipy.optimize import minimize, approx_fprime
from uncertainties import unumpy as unp
import states_and_witnesses as states
import operations as op

def get_nom(params, expec_vals, func):
            '''For use in error propagation; returns the nominal value of the function'''
            w = func(params, expec_vals)
            return unp.nominal_values(w)

# Witness scipy minimization functions
def min_W(x0, expt, bounds, counts, W):
    '''
    Returns a scipy object that has the function that gets minimized
    and the params used to minimize
    '''
    if expt:
        arr = get_nom
        args = (counts, W)
    else:
        arr = W
        args = (counts,)
    return minimize(arr, x0=x0, args=args, bounds=bounds)

def min_W_val(x0, expt, bounds, counts, W):
            # returns minimum expectation value of W
            return min_W(x0, expt, bounds, counts, W).fun
def min_W_params(x0, expt, bounds, counts, W):
            # returns the parameters that got minimized
            return min_W(x0, expt, bounds, counts, W).x



class InitGuess:
    """
    An object class that handles initial guesses and automatically
    uses scipy to minimize them

    Params:
        num_params: the number of params to minimize (theta, alpha, beta)
        expt: whether the input is experimental data
        counts: given if we want to just use raw counts, or calculate experimental rho
        W: the witness to be minimized

    Attributes:
        self.bounds: an array that holds the bounds for each randomized position component
        self.x0: an array that stores the initial position/conditions of the guess
            self.x0[0]: theta
            self.x0[1]: alpha (if it exists)
            self.x0[2]: beta (if it exists)
        self.w_min_val: the minimum expectation value of W for this guess
        self.w_min_params: the parameters we used to get that minimized W expectation value
    """
    def __init__(self, num_params, expt, counts, W):
        if num_params == 1:
            self.bounds = [(0, np.pi)]
            self.x0 = [np.random.rand()*np.pi]
        elif num_params == 3:
            self.bounds = [(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)]
            self.x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
        else: 
            self.bounds = [(0, np.pi/2),(0, np.pi*2)]
            self.x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
        # using scipy minimization
        self.w_min_val = min_W_val(self.x0, expt, self.bounds, counts, W)
        self.w_min_params = min_W_params(self.x0, expt, self.bounds, counts, W)

    # "magic methods" help us compare instances of a class
    def __eq__(self, other):
        return self.w_val == other.w_val
    def __ne__(self, other):
        return self.w_val != other.w_val
    def __lt__(self, other):
        return self.w_val < other.w_val
    def __le__(self, other):
        return self.w_val <= other.w_val
    def __gt__(self, other):
        return self.w_val > other.w_val
    def __ge__(self, other):
        return self.w_val >= other.w_val
    
# TODO: Look into sklearn SGDRegressor or tensorflow
def gradient_descent(guess, num_params, expt, counts, W, zeta=0.7, num_reps = 50):
    """
    Does gradient descent optimization on a given minimized guess

    Params:
        guess: an InitGuess object that has already been minimized with scipy
        params_arr: generates a new random guess based on how many params the W has
        bounds: ensures min_W_val is called with the right bounds based on number of params for W
        zeta: learning rate 
        num_reps: int, number of times to run the optimization
    Returns:
        guess: an InitGuess object that represents the optimized W
    """
    x0 = guess.x0
    w_min_val = guess.w_min_val
    w_min_params = guess.w_min_params
    bounds = guess.bounds
    isi = 0 # index since last improvement
    for _ in range(num_reps): # repeat 10 times and take the minimum
        if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
            new_guess = InitGuess(num_params, expt, counts, W)
            x0 = new_guess.x0
        else:
            grad = approx_fprime(guess.x0, min_W_val, 1e-6)
            if np.all(grad < 1e-5*np.ones(len(grad))):
                break
            else:
                x0 = x0 - zeta*grad

        w_val = min_W_val(x0, expt, bounds, counts, W)
        w_params = min_W_params(x0, expt, bounds, counts, W)
                        
        if w_val < w_min_val:
            w_min_val = w_val
            w_min_params = w_params
            isi=0
        else:
            isi+=1
    
    # update attributes of the guess based on minimization
    guess.x0 = x0
    guess.w_min_val = w_min_val
    guess.w_min_params = w_min_params
    return guess

def optimize_Ws(all_W, expt, counts, return_params):
    """
    Minimizes all of the witness expectation values using scipy and gradient descent

    Params:
        all_W: the list of all Ws and W's to minimize
        expt: whether the input is experimental data
        counts: given if we want to just use raw counts, or calculate experimental rho
        return_params: whether we want to return the params we used to get minimized Ws
    Returns:
        (W_expec_vals, min_params)
        W_expec_vals: the list of all minimized W expectation values,
        min_params: the list of all params used to get minimized Ws
    """
    W_expec_vals = []
    min_params = []

    # break into three groups based on the number of params to optimize
    for i, W in enumerate(all_W):
        if i <= 5: # These Ws only have theta, so just optimize theta
            num_params = 1
            # Try three different starting conditions (initial guesses)
            # Guesses are automatically minimized using scipy
            x0 = InitGuess(num_params, expt, counts, W)
            x1 = InitGuess(num_params, expt, counts, W)
            x2 = InitGuess(num_params, expt, counts, W)
            
            # The best guess out of these has the smallest minimized starting condition
            best_guess = min(x0, x1, x2)

            # Use best initial guess in Gradient descent to minimze Ws further
            minimized_W = gradient_descent(best_guess, num_params, expt, counts, W)
        
        # These witnesses have three parameters to be minimized (theta, alpha, and beta)
        elif i==8 or i==11 or i==14:
            num_params = 3
            # Two different starting conditions (init. guesses)
            # Process is the same as the one before (just with different parameters)
            x0 = InitGuess(num_params, expt, counts, W)
            x1 = InitGuess(num_params, expt, counts, W)

            best_guess = min(x0, x1)
            minimized_W = gradient_descent(best_guess, num_params, expt, counts, W)
            
        # The rest of the witnesses have 2 parameters to minimize (theta and alpha)
        else:
            num_params = 2   
            # Two init. guesses
            x0 = InitGuess(num_params, expt, counts, W)
            x1 = InitGuess(num_params, expt, counts, W)

            best_guess = min(x0, x1)
            minimized_W = gradient_descent(best_guess, num_params, expt, counts, W)

        if expt: # automatically calculate uncertainty
            # TODO: what/where is def W()?
            W_expec_vals.append(W(minimized_W.w_min_params))
        if return_params:
            min_params.append(minimized_W.w_min_params)
        else:
            W_expec_vals.append(minimized_W.w_min_val)
        return (W_expec_vals, min_params)

def compute_witnesses(rho, counts = None, expt = False, verbose = True, do_counts = False, 
                      expt_purity = None, model=None, optimize = True, gd=True, ads_test=False, return_all=False, 
                      return_params=False):
    ''' Computes the minimum of the 6 Ws and the minimum of the 3 triples of the 9 W's. 
        Params:
            rho: the density matrix
            counts: raw np array of photon counts and uncertainties
            -------expt: bool, whether to compute the Ws assuming input is experimental data
            verbose: Whether to return which W/W' are minimal.
            do_stokes: bool, whether to compute stokes parameters
            -------do_counts: use the raw definition in terms of counts 
            expt_purity: the experimental purity of the state, which defines the noise level: 1 - purity.
            ?model: which model to correct for noise; see det_noise in process_expt.py for more info
            model_path: path to noise model csvs.

            # rewrite
            optimize: bool, whether to optimize the Ws or to just check bounds
            gd: bool, whether to use gradient descent or brute random search

            ?ads_test: bool, whether to return w2 expec and sin (theta) for the amplitude damped states
            return_all: bool, whether to return all the Ws or just the min of the 6 and the min of the 3 triples
            return_params: bool, whether to return the params that give the min of the 6 and the min of the 3 triples
    '''
    # check if experimental data
    if expt and counts is not None:
        do_counts = True
    # if wanting to account for experimental purity, add noise to the density matrix for adjusted theoretical purity calculation

    # With experimental data
    if do_counts:
        counts = np.reshape(counts, (36,1))
        HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts

        def get_W1(params):
            a, b = np.cos(params), np.sin(params)
            # each element in counts is of form: count +/ uncertainty
            return np.real(0.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W2(params):
            a, b = np.cos(params), np.sin(params)
            return np.real(0.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W3(params):
            a, b = np.cos(params), np.sin(params)
            return np.real(0.25*(1 + ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W4(params):
            a, b = np.cos(params), np.sin(params)
            return np.real(0.25*(1 - ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) - 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W5(params):
            a, b = np.cos(params), np.sin(params)
            return np.real(0.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_W6(params):
            a, b = np.cos(params), np.sin(params)
            return np.real(0.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        
        ## W' from summer 2022 ##
        def get_Wp1(params):
            theta, alpha = params[0], params[1]
            return np.real(.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))+((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) - ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp2(params):
            theta, alpha = params[0], params[1]
            return np.real(.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))-((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) - np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) + ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp3(params):
            theta, alpha, beta = params[0], params[1], params[2]
            return np.real(.25 * (np.cos(theta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(theta)**2*np.cos(2*alpha - beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*((DD + DA - AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.cos(alpha - beta)*((DD - DA + AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.sin(alpha)*((RR - LR + RL - LL) / (RR + LR + RL + LL)) + np.sin(2*theta)*np.sin(alpha - beta)*((RR + LR - RL - LL) / (RR + LR + RL + LL))+np.cos(theta)**2*np.sin(beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) - ((DR - DL - AR + AL) / (DR + DL + AR + AL))) + np.sin(theta)**2*np.sin(2*alpha - beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) + ((DR - DL - AR + AL) / (DR + DL + AR + AL)))))
        def get_Wp4(params):
            theta, alpha = params[0], params[1]
            return np.real(.25*(1+((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((DD + DA - AD - AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp5(params):
            theta, alpha = params[0], params[1]
            return np.real(.25*(1-((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((DD + DA - AD - AA) / (DD + DA + AD + AA))) - np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp6(params):
            theta, alpha, beta = params[0], params[1], params[2]
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) + ((RR - LR + RL - LL) / (RR + LR + RL + LL))) + np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((RR - LR + RL - LL) / (RR + LR + RL + LL))) - np.cos(theta)**2*np.sin(2*alpha)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL))) - np.sin(theta)**2*np.sin(2*beta)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp7(params):
            theta, alpha = params[0], params[1]
            return np.real(.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV))) - np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL))+((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp8(params):
            theta, alpha = params[0], params[1]
            return np.real(.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV))-((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA))+((DH - DV - AH + AV) / (DH + DV + AH + AV)))+np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp9(params):
            theta, alpha, beta = params[0], params[1], params[2]
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.cos(theta)**2*np.sin(2*alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(theta)**2*np.sin(2*beta)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DH - DV - AH + AV) / (DH + DV + AH + AV)))+ np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV)))))

        # Optimization
        all_W = [get_W1, get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
        (W_expec_vals, min_params) = optimize_Ws(all_W, expt, counts, return_params)
        W_min = np.real(min(W_expec_vals[:6]))
        
        try:
            Wp_t1 = np.real(min(W_expec_vals[6:9])[0])
            Wp_t2 = np.real(min(W_expec_vals[9:12])[0])
            Wp_t3 = np.real(min(W_expec_vals[12:15])[0])
        except TypeError:
            Wp_t1 = np.real(min(W_expec_vals[6:9]))
            Wp_t2 = np.real(min(W_expec_vals[9:12]))
            Wp_t3 = np.real(min(W_expec_vals[12:15]))
        
        # For testing and for specific cases
        if verbose:
            #print('i got to verbosity')
            # Define dictionary to get name of
            all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9']
            index_names = {i: name for i, name in enumerate(all_W)}
           
            W_param = [x for _,x in sorted(zip(W_expec_vals[:6], min_params[:6]))][0]
            Wp_t1_param = [x for _,x in sorted(zip(W_expec_vals[6:9], min_params[6:9]))][0]
            Wp_t2_param = [x for _,x in sorted(zip(W_expec_vals[9:12], min_params[9:12]))][0]
            Wp_t3_param = [x for _,x in sorted(zip(W_expec_vals[12:15], min_params[12:15]))][0]
           
           
            W_exp_val_ls = []
            for val in W_expec_vals:
                W_exp_val_ls.append(unp.nominal_values(val))
            
            W_min_name = [x for _,x in sorted(zip(W_exp_val_ls[:6], all_W[:6]))][0]
            Wp1_min_name = [x for _,x in sorted(zip(W_exp_val_ls[6:9], all_W[6:9]))][0]
            Wp2_min_name = [x for _,x in sorted(zip(W_exp_val_ls[9:12], all_W[9:12]))][0]
            Wp3_min_name = [x for _,x in sorted(zip(W_exp_val_ls[12:15], all_W[12:15]))][0]
            
            print('Wp2 and its params are:', W_expec_vals[7], min_params[7])
            print('The found W and param are:', Wp_t1, Wp1_min_name, Wp_t1_param)

            if not return_params:
                return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name
            else:
                # return same as above but with the minimum params list at end
                return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                
        else:
            return W_min, Wp_t1, Wp_t2, Wp_t3
        
        # return W_expec_vals

    # Using theoretical data
    else: # use operators instead like in eritas's matlab code
        # bell states
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
            W = phi * op.adjoint(phi)
            W = op.partial_transpose(W) # take partial transpose
            return np.real(np.trace(W @ rho)) # minimizing this gives the expectation value of witness
        
        
        def get_W_matrix(state):
            return op.partial_transpose(state * op.adjoint(state))

        
        # Only difference for witnesses is how they are calculated

        ## ------ for W ------ ##
        def get_W1(param):
            a,b = np.cos(param), np.sin(param)
            phi1 = a*PHI_P + b*PHI_M
            return get_W_matrix(phi1)
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

        if not(ads_test):
            W_expec_vals = []
            if return_params: # to log the params
                min_params = []
            
            # Same optimization (essentially) as for experimental
            all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
            (W_expec_vals, min_params) = optimize_Ws(all_W, expt, counts, return_params)
            W_min = np.real(min(W_expec_vals[:6]))
            # print('W', np.round(W_expec_vals[:6], 3))
            # print('W\'', np.round(W_expec_vals[6:], 3))
            # find min witness expectation values
            W_min = min(W_expec_vals[:6])
            Wp_t1 = min(W_expec_vals[6:9])
            Wp_t2 = min(W_expec_vals[9:12])
            Wp_t3 = min(W_expec_vals[12:15])
            # get the corresponding parameters
            if return_params:
                W_expec_vals_ls = []
                for val in W_expec_vals:
                    W_expec_vals_ls.append(unp.nominal_values(val))
                # sort by witness value; want the most negative, so take first element in sorted
                W_param = [x for _,x in sorted(zip(W_expec_vals_ls[:6], min_params[:6]))][0]
                Wp_t1_param = [x for _,x in sorted(zip(W_expec_vals_ls[6:9], min_params[6:9]))][0]
                Wp_t2_param = [x for _,x in sorted(zip(W_expec_vals_ls[9:12], min_params[9:12]))][0]
                Wp_t3_param = [x for _,x in sorted(zip(W_expec_vals_ls[12:15], min_params[12:15]))][0]


            if not(return_all):
                if verbose:
                    #print('i got to verbosity')
                    # Define dictionary to get name of
                    all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9']
                    index_names = {i: name for i, name in enumerate(all_W)}
                
                    W_exp_val_ls = []
                    for val in W_expec_vals:
                        W_exp_val_ls.append(unp.nominal_values(val))
                    
                   
                    W_min_name = [x for _,x in sorted(zip(W_expec_vals[:6], all_W[:6]))][0]
                    Wp1_min_name = [x for _,x in sorted(zip(W_expec_vals[6:9], all_W[6:9]))][0]
                    Wp2_min_name = [x for _,x in sorted(zip(W_expec_vals[9:12], all_W[9:12]))][0]
                    Wp3_min_name = [x for _,x in sorted(zip(W_expec_vals[12:15], all_W[12:15]))][0]

                    if not return_params:
                        # Find names from dictionary and return them and their values
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name
                    else:
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                if return_params:
                    return W_min, Wp_t1, Wp_t2, Wp_t3, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                else:
                    return W_min, Wp_t1, Wp_t2, Wp_t3
            else:
                if return_params:
                    return W_expec_vals, min_params
                else:
                    return W_expec_vals
        else: 
            print('i went to the 2nd else')
            W2_main= minimize(get_W2, x0=[0], bounds=[(0, np.pi)])
            W2_val = W2_main['fun']
            W2_param = W2_main['x']

            return W2_val, W2_param[0]


def test_witnesses():
    '''Calculate witness vals for select experimental states'''
    r1 = np.load("../../framework/decomp_test/rho_('E0', (45.0, 18.0))_32.npy", allow_pickle=True)
    counts1 = unp.uarray(r1[3], r1[4])
    print('45, 18, 32')
    print(counts1)
    print(compute_witnesses(r1[0], counts1, expt=True))
    print('------')
    r2 = np.load("../../framework/decomp_test/rho_('E0', (59.99999999999999, 72.0))_32.npy", allow_pickle=True)
    counts2 = unp.uarray(r2[3], r2[4])
    print('60, 72, 32')
    print(counts2)
    print(compute_witnesses(r2[0], counts2, expt=True))
    print('------')
# file to generate systems by enforcing orthogonality of measured states
import numpy as np
import time
from scipy.special import comb

class HyperBell():
    '''HyperBell class to generate systems of measured bell states for a given dimension d and number of states in group k.'''

    def init(self, d, k):
        '''Initialize HyperBell class with dimension d and number of states in group k.'''
        self.d = d
        self.k = k

        print('Initializing HyperBell class with d = {} and k = {}.'.format(d,k))

        self.soln_limit = 2*d # must find solutions for all detectors to fuflill sufficient condition

        self.num_coeff = 2*d # number of coefficients in measurement operator

        self.m_limit = comb(d**2, k)

        # self.n_bounds = [(0, self.d) for _ in range(self.num_coeff)] # set bounds for m
        # self.q_bounds = [(0, self.d-1) for _ in range(self.num_coeff)] # set bounds for q
        # self.bounds = self.n_bounds + self.q_bounds # set bounds for m and q
        self.bounds = [(-1, 1) for _ in range(self.num_coeff)]

        self.precision = 6 # precision for solution vec, orthoginality, and normalization

        # self.nq_precision = 3 # precision for rounding up n and q values

        self.start_time = time.time() # start timer

        self.set_m(0)

    def get_bell(self, c, p):
        '''Define bell state for given correlation and phase class c, p and dimension d, represented in number basis.
        --
        Params: 
            c (int): correlation class
            p (int): phase class
        --
        Returns:
            bell (np.array): bell state in number basis, 2d x1 vector
        '''
        d = self.d
        bell = np.zeros((2*d,1), dtype='complex128')
        for l in range(d):
            phase = np.exp(1j*2*np.pi * p *l / d)
            numb = np.zeros((2*d,1)) # number basis portion
            numb[l]=1
            numb[d+(l+c)%d]=1
            bell += phase * numb
        return bell * 1/np.sqrt(d) # normalize

    def get_m_k_ind(self):
        '''Returns the mth combination of d^2 choose k elements; uses CNS (combinaitorial number system)'''
        m= self.m
        d = self.d
        k = self.k
        if m > comb(d**2, k):
            raise ValueError(f'Must have m in comb(d^2, k). You have m={m}')
        def do_round(targ, n):
            '''Funds the largest value of d^2 choose n to targ'''
            if targ==0:
                largest_val = 0
                ind = n-1;
                return largest_val, ind
            elif targ==1:
                largest_val = 0
                ind = n
                return largest_val, ind
            i=-1
            largest_val = 0
            while largest_val <= targ and n+1+i <= d**2:
                largest_val = comb(n+1+i, n)
                if largest_val > targ:
                    i-=1
                    ind = n+i+1
                    if n+1+i == n-1:
                        largest_val = 0
                        ind = n-1
                    else:
                        largest_val = comb(n+1+i, n)
                        ind = n+1+i
                    return largest_val, ind
            return largest_val, ind
        # targ starts at m
        targ = m
        m_k_ind = []
        for j in range(k, 0, -1):
            largest_val, ind = do_round(targ, j)
            m_k_ind.append(ind)
            targ = abs(targ-largest_val)
        return m_k_ind

    def get_m_kgroup(self):
        '''Returns mth k-group of bell states.'''
        k_group_indices = self.get_m_k_ind()
        k_group = []
        for i in k_group_indices:
            c = i // self.d
            p = i % self.d
            k_group.append(self.get_bell(c,p))
        return k_group

    def get_one_coeff(self, nq):
        '''Returns input coefficients for measurement operator.
        Params:
            nq (tuple): (n,q) tuple of integers; form of exact solution is sqrt(n) / sqrt(d) *e^(2pi i q / d)
        
        '''
        n, q = nq
        d = self.d
        # m = np.sqrt(n)
        return n / d**2 * np.exp(2*np.pi*1j*q/d)  

    def get_coeff(self, nq_ls):
        '''Returns input coefficients for measurement operator.
            ---
            Params:
                nq_ls (np.array): array of integers: first 2d are n, second 2d are q
        '''
        d = self.d
        coeff = []
        for i in range(len(nq_ls)//2):
            coeff.append(self.get_one_coeff((nq_ls[i], nq_ls[2*d+i])))
        return np.array(coeff)

    def get_meas(self, bell):
        '''Performs LELM measurement on bell state b.
        --
        Params:
            bell (np.array): bell state in number basis, 2d x1 vector
        --
        Returns:
            meas (func): returns an array taking as input coefficients which in turn return the measured state in number basis, 2d x1 vector; note coefficients are split into real and imaginary parts to make solving process easier
        --
        '''
        d = self.d
        # define measurement coeffients
        def meas(coeff):
            bell_m = np.zeros((2*d,1), dtype='complex128') # initialize measured bell state
            for l in range(2*d):
                if bell[l]!=0: # if state is occupied
                    bell_c = bell.copy()
                    bell_c[l]=0 # annihilate state
                    bell_m += coeff[l] * bell_c # mutliply by coefficient
            return bell_m

        return meas

    def get_norm(self, coeff):
        '''Computes norm of input vector.'''
        return np.sqrt(sum([abs(coeff[i])**2 for i in range(len(coeff))]))

    def get_meas_ip(self, bell_ls, coeff):
        ''' Computes inner product between two measured bell states'''
        # unpack bell states ls
        try:
            bell1, bell2 = bell_ls
        except:
            print(bell_ls)
        # get measurement functions
        bell1_meas_func = self.get_meas(bell1)
        bell2_meas_func = self.get_meas(bell2)
        # get measured states
        bell1_meas = bell1_meas_func(coeff)
        bell2_meas = bell2_meas_func(coeff)
        # print(bell1_meas)
        # print(bell2_meas)
        # compute inner product
        ip = np.conj(bell1_meas).T @ bell2_meas
        ip = ip[0][0] # extract scalar from array
        return ip

    def is_valid_soln(self, coeff):
        '''Checks if a single solution is valid.'''
        # check if solution is valid
        valid_soln = np.all(np.round(self.get_ksys(coeff), self.precision) == 0) and np.round(self.get_norm(coeff), self.precision) == 1 # check if solution is valid
        # if not(valid_soln):
        #     print('invalid soln')
        #     print('soln vector', self.get_ksys(coeff))
        #     print('soln x = ', coeff)
        #     print('norm = ', self.get_norm(coeff))
        return valid_soln

    def is_orthogonal(self, coeff, coeff_ls):
        '''Checks if a single solution is orthogonal to all existing solutions.'''
        valid_soln = True
        in_coeff_ls = False
        for x in coeff_ls:
            if not(np.all(coeff == x)):
                ip = x.conj().T @ coeff
                try:
                    ip = ip[0][0]
                except IndexError:
                    pass
                    ip = np.real(ip)
                if np.round(ip, self.precision) != 0:
                    valid_soln = False
                    break
            else:
                in_coeff_ls = True
        return valid_soln, in_coeff_ls

    def set_m(self, m):
        '''Sets m to be the number of the k-system under investigation.'''
        self.m = m

    def set_num_attempts(self, num_attempts):
        '''Sets number of attempts to find solutions.'''
        self.num_attempts = num_attempts

    def set_precision(self, precision):
        '''Sets precision as the negative exponent for declaring solutions in terms of 
            - func vector = 0
            - inner product = 0
            - normalization = 1
        '''
        self.precision = precision

    def get_ksys(self, coeff):
        '''Function to return the mth k-system as a function of 4*d real coefficients.
        --
        Params:
            i (int): index of system in 
            coeff (np.array, 4*d x 1): coefficients for measurement
        '''
        k_group = self.get_m_kgroup() # get ith k-group
        # take inner product of all pairs of k bell states
        k_sys = [] # list of all inner products for a given group, which make up a system
        for i in range(len(k_group)):
            for j in range(i+1,len(k_group)):
                k_sys.append(self.get_meas_ip(coeff=coeff, bell_ls = [k_group[i], k_group[j]]))
        return np.array(k_sys)

    def get_allsys_func(self):
        '''Returns a vector of all measured states as a function of 4*d real coefficients.
        --
        Params:
            d (int): dimension of system
            k (int): number of states in group
            coeff (np.array, 4*d x 1): coefficients for measurement
        '''
        all_sys = []
        for i in range(len(self.k_groups)):
            all_sys.append(self.get_ksys(i))
        return np.array(all_sys)

    def rand_guess(self):
        '''Random stick simplex to guess input vector'''
        rand = np.random.rand(self.num_coeff-1)
        rand = np.sort(rand)
        guess = np.zeros(self.num_coeff)
        guess[0] = rand[0]
        for i in range(1, len(rand), 1):
            guess[i] = rand[i] - rand[i-1]
        guess[-1] = 1 - rand[-1]
        # extend the simplex into the negatives
        for i in range(len(guess)):
            n = np.random.rand() < 0.5
            if n:
                guess[i] *= -1
        return guess

    # def rand_guess(self):
    #     '''Returns random guess for m, q val.'''
    #     # random stick for m, guess random q
    #     rand = np.random.randint(low = 0, high= self.d+1, size=(self.num_coeff-1))
    #     rand = np.sort(rand)
    #     n_guess = np.zeros(self.num_coeff)
    #     n_guess[0] = rand[0]
    #     for i in range(1, len(rand), 1):
    #         n_guess[i] = rand[i] + rand[i-1]
    #     n_guess[-1] = self.d- rand[-1]
    #     # random guess for q
    #     q_guess = np.random.randint(0, self.d, size=(self.num_coeff))
    #     return np.concatenate((n_guess, q_guess))

   

if __name__ == '__main__':
    # test
    d = 2
    k = 3
    hb = HyperBell()
    hb.init(d,k)
    coeff=np.array([-0.9496828 , -0.84160771, -0.53092842, -0.83225099, -0.90072945,
       -0.35618569, -0.86866719, -0.71994774])
    print(hb.get_meas(hb.get_bell(c=0, p=0))(coeff))
    print(hb.get_ksys(coeff))
    print(hb.get_ksys(hb.rand_guess()))
    # print(hb.get_ksys_func(0)())
    # print(hb.get_ksys(0, hb.rand_guess()))

    def meas(bell, coeff):
        bell_m = np.zeros((2*d,1), dtype='complex128') # initialize measured bell state
        print(bell)
        for l in range(2*d):
            if bell[l]!=0: # if state is occupied
                bell_c = bell.copy()
                bell_c[l]=0 # annihilate state
                bell_m += (coeff[l] + 1j*coeff[2*d+l]) * bell_c # mutliply by coefficient
        print(bell_m)
        return bell_m
    meas(hb.get_bell(c=0, p=0), coeff)


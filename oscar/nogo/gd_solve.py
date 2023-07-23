# file to solve distinguisability systems numerically using gradient descent optimization on the initally guesses to the fsolve function
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import time
from os.path import join
import matplotlib.pyplot as plt

from solve_prep import HyperBell # import HyperBell class to handle system generation

def solve_ksys(m, hb, verbose=True, opt=False):
    '''Function to solve the mth k-system using gradient descent optimization on the initally guesses to the fsolve function.
    --
    Params: 
        i (int): index of k-system to solve
        hb (HyperBell): HyperBell class object
        verbose: whether to update progress per individual search, bool
        opt: whether this solve is for parallel or single process, bool
    '''
    
    hb.set_m(m) # set mth k-system

    def loss(x0, x_ls = None): 
        '''Compute loss function for given coefficients x0. Three parts: 
            l0: absolute difference in norm from 1
            l1: RSE for function values
            l2: RSE for inner products with existing solutions'''

        l0 = abs(1-hb.get_norm(x0))

        l1 =  np.sqrt(sum([abs(hb.get_ksys(x0)[i])**2 for i in range(len(hb.get_ksys(x0)))]))

        if x_ls is not None:
            ip_ls = []
            for x in x_ls:
                if not(np.all(x0 == x)):
                    ip = x.conj().T @ x0
                    try:
                        ip = ip[0][0]
                    except IndexError:
                        pass
                    ip = abs(ip)
                    ip_ls.append(ip)
            l2 = np.sqrt(sum([abs(ip_ls[i])**2 for i in range(len(ip_ls))]))
            return l0 + l1 + l2
        else:
            return l0 + l1

    def get_soln(x0, x_ls):
        '''Attempt to find single solution of system with given initial guess x0.'''
        def try_solve(x0):
            soln= minimize(loss, args=(x_ls,), x0 = x0, bounds=hb.bounds)
            s_loss = soln.fun
            # create tuple of (n,q) values
            # nq_sol = soln.x
            # nq_sol = np.round(nq_sol, hb.nq_precision)
            # nq_sol_r = np.reshape(nq_sol, (2,hb.num_coeff))
            # n_ls = nq_sol_r[0, :]
            # q_ls = nq_sol_r[1, :]
            # nq_sol_r = np.array(list(zip(n_ls, q_ls)))

            x_sol = soln.x
            
            valid_soln = hb.is_valid_soln(x_sol)# check if solution is valid

            soln_vec = hb.get_ksys(x_sol)
            soln_vec = np.round(soln_vec, hb.precision)
        
            return s_loss, soln_vec, x_sol, valid_soln

        s_loss, soln_vec, x_sol, valid_soln = try_solve(x0)
        it = 0
        while not(valid_soln):
            x0 = hb.rand_guess()
            # print('not valid soln, using x0=', x0)
            # print('coeff', x_sol)
            # print('soln_vec', soln_vec)
            # print('loss', s_loss)
            
            # if verbose:
            #     print('trying guess x0', x0)
            s_loss, soln_vec, x_sol, valid_soln = try_solve(x0)
            it+=1

        if valid_soln:
            if verbose: 
                print('found soln! Loss: ', s_loss)
                print('soln vector', soln_vec)
                print('soln x = ', x_sol)
                # print('(m,q) soln = ',nq_sol)
            return soln_vec, x_sol, it

    # initializing
    x_ls = []
    # nq_ls = [] # tuples of (n,q) values
    soln_ls = [] # list of solution vectors
    it_ls = [] # list of number of iterations required to find a valid solution
    s = 0
    # get initial guess
    x0 = hb.rand_guess()
    while (len(x_ls) < hb.soln_limit) and (s < hb.soln_limit): 
        soln_vec, x_sol, it = get_soln(x0, x_ls)
        if len(x_ls) > 0:
            orthogonal, in_x_ls = hb.is_orthogonal(x_sol, x_ls)
            if orthogonal and not(in_x_ls):
                print('adding to x_ls!')
                x_ls.append(x_sol)
                # nq_ls.append(nq_sol)
                soln_ls.append(soln_vec)
                it_ls.append(it)
                s+=1 # increment number of successful attempts
        else:
            print('first solution!')
            x_ls.append(x_sol)
            # nq_ls.append(nq_sol)
            soln_ls.append(soln_vec)
            it_ls.append(it)
            s+=1 # increment number of successful attempts
        
        x0 = hb.rand_guess()

    if len(x_ls) == hb.soln_limit:
        if verbose: print('found all solns with m = %i, d = %i, k = %i'%(m,hb.d, hb.k ))
        if opt: 
            print('x_ls', x_ls)
            # print('nq_ls', nq_ls)

            print('saving!')
            # nq_ls = np.array(nq_ls)
            x_ls = np.array(x_ls)
            soln_ls = np.array(soln_ls)
            it_ls = np.array(it_ls)
            # print('nq_ls shape', nq_ls.shape)
            # print('x_ls shape', x_ls.shape)
            # print('soln_ls shape', soln_ls.shape)
            # print('it_ls shape', it_ls.shape)
            np.savez(join('results','solns_d%i_k%i_m%i_p%i_nqp%i.npz'%(hb.d, hb.k, m, hb.precision, hb.nq_precision)), array1=x_ls, array2=soln_ls, array3=it_ls)

            # make histogram of number of iterations
            plt.figure(figsize=(10,5))
            plt.hist(it_ls)
            plt.xlabel('Number of Iterations')
            plt.ylabel('Frequency')
            plt.title('Finding Valid Solution for $m = %i, d = %i, k = %i$'%(m, hb.d, hb.k))
            plt.savefig(join('results','it_hist_d%i_k%i_m%i_p%i_nqp%i.pdf'%(hb.d, hb.k, m, hb.precision, hb.nq_precision)))

            tf = time.time()
            print('time elapsed: ', tf-hb.start_time)

            raise StopAsyncIteration('found all solns with m = ', m)

        return x_ls
    else:
        if verbose: 
            print('did not find all solns with m = %i, d = %i, k = %i'%(m, hb.d, hb.k ))
            print('found %i solns'%len(x_ls)) 
            tf = time.time()
            print('time elapsed: ', tf-hb.start_time) 
        return x_ls
                
def solve_all_ksys(hb, verbose=True):
    '''Function to solve all k-systems using gradient descent optimization on the initally guesses to the fsolve function.
    --
    Params: 
        hb (HyperBell): HyperBell class object
    '''
    ## initialize parallelization ##
    pool = Pool(cpu_count())
    solve_ksys_set = partial(solve_ksys, hb=hb,verbose=verbose, opt=True)
    inputs = [m for m in range(int(hb.m_limit))]
    try:
       # Process the data using the multiprocessing pool
        for result in pool.imap_unordered(solve_ksys_set, inputs):
            print(result)  # Print the result


    except StopIteration as e:
        print("Condition satisfied:", e)

   
    pool.close()
    pool.join()


if __name__ == '__main__':
    # define d and k vals
    d= int(input('Enter d: '))
    k = int(input('Enter k: '))
    # initialize HyperBell class
    hb = HyperBell() 
    hb.init(d, k) # starts timer

    # N_fac = int(input('What multiple of solution limit to run for: '))
    # hb.set_num_attempts(N_fac*hb.soln_limit)
    # precision = int(input('Enter precision for func soln and inner products as negative of exponent (e.g., 9 for 10^-9): '))
    # hb.set_precision(precision)
    # not0_precision = int(input('Enter precision for how far away from 0 >=1 value in input must be as negative of exponent (e.g., 9 for 10^-9): '))
    # hb.set_not0_precision(not0_precision)

    # solve all k-systems
    
    solns = solve_all_ksys(hb)
    # else:
    #     m_ls = range(hb.num_ksys)
    #     for m in tqdm(m_ls):
    #         solns = solve_ksys(m=m, hb = hb, opt=False)
    #         if len(solns) >= hb.soln_limit:
    #             print('found %i solns for k-system %i'%(len(solns), m))
    #             print('solns', solns)
    #             solns = np.array(solns)
    #             np.save(join('results', 'solns_d%i_k%i_m%i_p%ipn%i_N%i.npy'%(hb.d, hb.k, m, hb.precision, hb.not0_precision, hb.num_attempts)), solns)
    #             break
    #         else:
    #             print('did not find all solns for k-system %i'%m)
# file to solve distinguisability systems numerically using gradient descent optimization on the initally guesses to the fsolve function
import numpy as np
from scipy.optimize import minimize, approx_fprime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import trange

from solve_prep import HyperBell # import HyperBell class to handle system generation

def solve_ksys(m, hb, zeta = 1, f=0.1, verbose=True):
    '''Function to solve the mth k-system using gradient descent optimization on the initally guesses to the fsolve function.
    --
    Params: 
        i (int): index of k-system to solve
        hb (HyperBell): HyperBell class object
        zeta: learning rate, float
        f: how often to break the GD and get random inputs, float
        N (int): number of iterations to run gradient descent
    '''
    N = 2*hb.soln_limit # set max number of iterations
    
    hb.set_m(m) # set mth k-system

    def loss(x0): 
        '''Compute loss function for given coefficients x0.'''
        return np.sqrt(sum([abs(hb.get_ksys(x0)[i])**2 for i in range(len(hb.get_ksys(x0)))]))

    bounds = [(0,1) for _ in range(4*hb.d)] # set bounds for coefficients

    # get initial guess
    x0 = hb.rand_guess()
    def get_soln(x0):
        # solve system
        def try_solve(x0):
            soln= minimize(loss, x0=x0, bounds=bounds)
            # print('x', soln.x)
            # print('fun', soln.fun)
            # print('x > 0', soln.x > 0)
            valid_soln = soln.success and np.isclose(soln.fun, 0, rtol=1e-6) and np.any(soln.x > 0)# check if solution is valid 
        
            return soln.fun, soln.x, valid_soln

        soln, x_sol, valid_soln = try_solve(x0)
        grad_x0= x0
        n = 0
        while not(valid_soln) and n <N:
            # if verbose:
            #     print(n, loss(x0))
            if n % (f * N)==0:
                x0 = hb.rand_guess()
            else:
                gradient = approx_fprime(grad_x0, loss, epsilon=1e-8)
                try:
                    x0 = [grad_x0[i] - zeta*gradient[i] for i in range(len(grad_x0))]
                    assert np.all(np.array(x0) > 0)
                except AssertionError:
                    x0 = hb.rand_guess()
            soln, x_sol, valid_soln = try_solve(x0)
            n+=1

        if valid_soln:
            if verbose: 
                print('found soln!', soln)
                print('given x0 = ', x0)
                print('soln x = ', x_sol)
            return soln, x_sol
        else:
            if verbose: 
                print('no soln found')
                print('given x0 = ', x0)
                print('soln x = ', x_sol)

            return None

    x_ls = []
    x_sol = np.zeros_like(x0)
    n = 0
    while (len(x_ls) < hb.soln_limit) and (n < N):
        try:
            soln, x_sol = get_soln(x0)
            print('soln', soln)
            print('num solns', len(x_ls))
            try:
                x_ls.index(x_sol) # check if soln is already in list; if yes, don't add
            except ValueError:
                print('already found this solution', x_sol)
                x_ls.append(x_sol)
        except TypeError:
            print('no soln found')
        n+=1
    if len(x_ls) == hb.soln_limit:
        if verbose: print('found all solns with m = ', m)
        # raise StopAsyncIteration('found all solns with m = ', m)
        return x_ls
    else:
        if verbose: print('did not find all solns')
        return x_ls
                
def solve_all_ksys(hb, N, zeta = 0.7, f=0.1, verbose=True):
    '''Function to solve all k-systems using gradient descent optimization on the initally guesses to the fsolve function.
    --
    Params: 
        hb (HyperBell): HyperBell class object
        zeta: learning rate, float
        f: how often to break the GD and get random inputs, float
        N (int): number of iterations to run gradient descent
    '''
    ## initialize parallelization ##
    pool = Pool(cpu_count())
    solve_ksys_set = partial(solve_ksys, hb=hb, N=N, zeta=zeta, f=f, verbose=verbose)
    inputs = [(m,) for m in range(hb.num_ksys)]
    try:
        results= pool.imap_unordered(solve_ksys_set, inputs)
    except StopAsyncIteration as e:
        print(e)
    finally:
        ## end multiprocessing ##
        pool.close()
        pool.join()

    # filter None results out
    results = [result for result in results if result is not None] 

    def check_solns(results):
        '''Check if all k-systems have been solved.'''
        return len(results) == hb.num_ksys
    
    if check_solns(results):
        print('found all solns for all k groups!')
        print(results)
        return results
    else:
        print('did not find all solns')
        return results

if __name__ == '__main__':
    # define d and k vals
    d = 6
    k = 7
    # initialize HyperBell class
    hb = HyperBell()
    hb.init(d, k)
    # solve all k-systems
    # solns = solve_all_ksys(hb)
    # for m in trange(hb.num_ksys):
    #     solns = solve_ksys(m, hb)
    #     if len(solns) >= hb.soln_limit:
    #         print('found %i solns for k-system %i'%(len(solns), m))
    #         break
    m_ls = range(hb.num_ksys)
    for m in m_ls:
        solns = solve_ksys(m, hb)
        if len(solns) >= hb.soln_limit:
            print('found %i solns for k-system %i'%(len(solns), m))
            print('solns', solns)
            break
        else:
            print('did not find all solns for k-system %i'%m)
            # break



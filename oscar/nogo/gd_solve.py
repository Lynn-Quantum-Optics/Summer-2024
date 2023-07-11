# file to solve distinguisability systems numerically using gradient descent optimization on the initally guesses to the fsolve function
import numpy as np
from scipy.optimize import minimize, approx_fprime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from os.path import join

from solve_prep import HyperBell # import HyperBell class to handle system generation

def solve_ksys(m, hb, zeta = 1, f=0.1, verbose=True, opt=False):
    '''Function to solve the mth k-system using gradient descent optimization on the initally guesses to the fsolve function.
    --
    Params: 
        i (int): index of k-system to solve
        hb (HyperBell): HyperBell class object
        zeta: learning rate, float
        f: how often to break the GD and get random inputs, float
        verbose: whether to update progress per individual search, bool
        opt: whether this solve is for parallel or single process, bool
    '''
    N = 10*hb.soln_limit # set max number of iterations
    
    hb.set_m(m) # set mth k-system

    def loss(x0, x_ls = None): 
        '''Compute loss function for given coefficients x0. Two parts: RSE for function values and RSE for inner products with existing solutions'''
        l1 =  np.sqrt(sum([abs(hb.get_ksys(x0)[i])**2 for i in range(len(hb.get_ksys(x0)))]))

        if x_ls is not None:
            v_sol = hb.construct_v(x0)
            ip_ls = []
            for x in x_ls:
                if not(np.all(x_sol == x)):
                    v_x = hb.construct_v(x)
                    ip = v_sol.conj().T @ v_x
                    ip = ip[0][0]
                    ip = np.real(ip)
                    ip_ls.append(ip)
            l2 = np.sqrt(sum([abs(ip_ls[i])**2 for i in range(len(ip_ls))]))
            return l1 + l2
        else:
            return l1

    bounds = hb.bounds # get bounds for coefficients

    # get initial guess
    x0 = hb.rand_guess()
    def get_soln(x0, x_ls):
        '''Attempt to find single solution of system with given initial guess x0.'''
        def try_solve(x0):
            soln= minimize(loss, args=(x_ls,), x0=x0, bounds=bounds)
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
                # gradient = approx_fprime(grad_x0, loss, epsilon=1e-8)
                # try:
                #     x0 = [grad_x0[i] - zeta*gradient[i] for i in range(len(grad_x0))]
                #     # x0 = hb.rand_guess()
                #     assert np.all(np.array(x0) > 0)
                # except AssertionError:
                #     x0 = hb.rand_guess()
                x0 = hb.rand_guess()
            soln, x_sol, valid_soln = try_solve(x0)
            n+=1

        if valid_soln:
            if verbose: 
                print('found soln! Loss: ', soln)
                print('soln vector', hb.get_ksys(x_sol))
                print('given x0 = ', x0)
                print('soln x = ', x_sol)
            return soln, x_sol
        else:
            if verbose: 
                print('no soln found')
                print('given x0 = ', x0)
                print('soln x = ', x_sol)

            return None

    # initializing
    x_ls = []
    n = 0
    while (len(x_ls) < hb.soln_limit) and (n < N): # try to find all solution
        try:
            soln, x_sol = get_soln(x0, x_ls)
            if len(x_ls) > 0 and soln is not None:
                             
                v_sol = hb.construct_v(x_sol)
                orthogonal = True
                in_x_ls = False
                print('x_ls', x_ls)
                for x in x_ls:
                    if not(np.all(x_sol == x)):
                        v_x = hb.construct_v(x)
                        ip = v_sol.conj().T @ v_x
                        try: ip = ip[0][0]
                        except IndexError: pass
                        ip = np.real(ip)
                        if ip > 1e-6: 
                            orthogonal= False # check if orthogonal
                            print('not orthogonal', ip)
                    else:
                        in_x_ls = True
                        break
                if orthogonal and not(in_x_ls):
                    print('adding to x_ls!')
                    x_ls.append(x_sol)
            
            else:
                print('first solution!')
                x_ls.append(x_sol)
        except TypeError:
            print('no soln found')
        n+=1
        x0 = hb.rand_guess()
    if len(x_ls) == hb.soln_limit:
        if verbose: print('found all solns with m = %i, d = %i, k = %i'%(m,hb.d, hb.k ))
        if opt: 
            print(x_ls)
            raise StopAsyncIteration('found all solns with m = ', m)
        return x_ls
    else:
        if verbose: 
            print('did not find all solns with m = %i, d = %i, k = %i'%(m, hb.d, hb.k ))
            print('found %i solns'%len(x_ls))  
        return x_ls
                
def solve_all_ksys(hb, zeta = 1, f=0.1, verbose=True):
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
    solve_ksys_set = partial(solve_ksys, hb=hb, zeta=zeta, f=f, verbose=verbose, opt=True)
    inputs = [m for m in range(hb.num_ksys)]
    try:
        # Process the data using the multiprocessing pool
        for result in pool.imap_unordered(solve_ksys_set, inputs):
            print(result)  # Print the result

    except StopIteration as e:
        print("Condition satisfied:", e)
    # except StopAsyncIteration as e:
    #     print(e)
    #     print(results)
    #     ## end multiprocessing ##
    #     pool.close()
    #     pool.join()
    #     return results
    # finally:
        ## end multiprocessing ##
    pool.close()
    pool.join()

    # filter None results out
    # results = [result for result in results if result is not None and len(result) > 0] 

    # def check_solns(results):
    #     '''Check if all k-systems have been solved.'''
    #     return len(results) == hb.num_ksys
    
    # if check_solns(results):
    #     print('found all solns for all k groups!')
    #     print(results)
    #     return results
    # else:
    #     print('could not distinguish')
    #     return results

if __name__ == '__main__':
    # define d and k vals
    d= int(input('Enter d: '))
    k = int(input('Enter k: '))
    # initialize HyperBell class
    hb = HyperBell()
    hb.init(d, k)
    # solve all k-systems
    opt = bool(int(input('Parallelize (1) or sequential (0): ')))
    if opt:
        solns = solve_all_ksys(hb)
    else:
        m_ls = range(hb.num_ksys)
        for m in tqdm(m_ls):
            solns = solve_ksys(m=m, hb = hb, opt=False)
            if len(solns) >= hb.soln_limit:
                print('found %i solns for k-system %i'%(len(solns), m))
                print('solns', solns)
                solns = np.array(solns)
                np.save(join('results', 'solns_d%i_k%i_m%i.npy'%(hb.d, hb.k, m)), solns)
                break
            else:
                print('did not find all solns for k-system %i'%m)
                # break
    # for m in trange(hb.num_ksys):
    #     solns = solve_ksys(m, hb)
    #     if len(solns) >= hb.soln_limit:
    #         print('found %i solns for k-system %i'%(len(solns), m))
    #         break
    #



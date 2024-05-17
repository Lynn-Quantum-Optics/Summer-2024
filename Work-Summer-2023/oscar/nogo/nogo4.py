# file to generate the systems in distinguishability problem
# theory: version 5/28/23

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.physics.quantum.dagger import Dagger
from itertools import *

## set d and k values ##
d = 3
k = 3

# initialize symbols
var_dict={}
vbet = [] # alphabet of vs
for i in range(2*d):
    v_i = Symbol('v'+str(i), complex=True)
    var_dict['v'+str(i)] = v_i
    vbet.append(v_i)

def measure_pair(cp1, cp2, pcondition):
    # define c and p
    c1, p1 = cp1[0], cp1[1]
    c2, p2 = cp2[0], cp2[1]

    # define our j1 and j2 indices
    j1_ls = np.arange(0, d, 1)
    j2_ls = np.arange(0, d, 1)

    eqn_ls = [] # variable to hold the system

    if c1 == c2:
        for j in np.arange(0, d, 1):
            # get params
            gamma = (j + c2) % d
            delta2 = 2*pi*I*p2*gamma / d
            delta1 = 2*pi*I*p1*gamma / d
            zeta = d + gamma
            phase = exp(delta2 - delta1)

            # compute terms
            term1 = phase*Dagger(var_dict['v'+str(j)])*var_dict['v'+str(j)] 
            term2 = phase*Dagger(var_dict['v'+str(zeta)])*var_dict['v'+str(zeta)]

            eqn_ls.append(term1)
            eqn_ls.append(term2)

    else:
        for j1 in j1_ls:
            for j2 in j2_ls:
                # get params
                gamma2 = (j2 + c2) % d
                gamma1 = (j1 + c1) % d
                delta2 = 2*pi*I*p2*gamma2 / d
                delta1 = 2*pi*I*p1*gamma1 / d

                # condition to determine the index
                if j1 == j2:
                    lambda1 = d + gamma1
                    lambda2 = d + gamma2
                elif gamma1 == gamma2:
                    lambda1 = j1
                    lambda2 = j2

                phase = exp(delta2 - delta1)
                term = phase*Dagger(var_dict['v'+str(lambda1)])*var_dict['v'+str(lambda2)]
                eqn_ls.append(term)

    eqn= sum(eqn_ls)
    def print_info(): # function to print out individual equations
        print(cp1, cp2)
        # print(latex(eqn))
        pprint(eqn)
        print('------')
    if pcondition:
        print_info()
    return eqn

## testing sample states ##
# eqn = measure_pair((0,0), (0,1), True)
eqn = measure_pair((0,1), (1,0), True)
# eqn = measure_pair((0,0), (1,1))


## for given k, figure out how many combinations ##
def build_cp():
    cp_ls = []
    for c in range(d):
        for p in range(d):
            cp_ls.append((c, p))
    return cp_ls

cp_ls= build_cp()
k_cp = list(combinations(cp_ls, k))

## perform measurements of d^2 choose k choose 2 pairs ##
def measure_group(k_cp, print_latex, print_normalorder):
    master_sys=[]
    for kg in k_cp:
        # have to get all combinations of pairs
        k_pair = list(combinations(kg, 2))
        sys= []

        if print_latex==True:
            print('For the $k=' + str(k) + '$ group $' + str(kg) +'$,')
            print('\\begin{equation}')
            print('\\begin{split}')
            for pair in k_pair:
                eqn = measure_pair(pair[0], pair[1], False)
                print(latex(eqn), " &= 0 \\\\")
                master_sys.append(eqn)
            print('\\end{split}')
            print('\\end{equation}')
        else:
            for pair in k_pair:
                eqn = measure_pair(pair[0], pair[1], False)
                sys.append(eqn)
            master_sys.append(sys)

        if print_normalorder==True:
            print('For the $k=' + str(k) + '$ group $' + str(kg) +'$,')
            print('\\begin{equation}')
            print('\\begin{split}')
            for pair in k_pair:
                eqn = measure_pair(pair[0], pair[1], False)
                cl = Poly(eqn).coeffs()[0] # get coefficient on first term, divide it out to standardize
                eqn /= cl
                print(latex(eqn), " &= 0 \\\\")
                master_sys.append(eqn)
            print('\\end{split}')
            print('\\end{equation}')

    return master_sys
# file to compute inner products on bell states, using 4/15/23 expression
# @oscar47

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.physics.quantum.dagger import Dagger
from itertools import *

## set d and k values ##
d = 2
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
            delta = d + ((d + j - c1) % d)
            phase = exp((2*pi * I* j)/d *(p2-p1))
            # phase = exp((2*pi * I* gamma)/d *(p2-p1))
            term1 = phase*Dagger(var_dict['v'+str(j)])*var_dict['v'+str(j)] 
            term2 = phase*Dagger(var_dict['v'+str(delta)])*var_dict['v'+str(delta)]
            # print(term1)
            eqn_ls.append(term1)
            eqn_ls.append(term2)

    else:
        for j1 in j1_ls:
            for j2 in j2_ls:
                gamma1 = d + ((d + j1 - c1) % d)
                gamma2 = d + ((d + j2 - c2) % d)
                if ((j1!=j2) and (c1 != c2)) or ((c1 == c2) and (j1 == j2)):
                    lambda1 = j1
                    lambda2 = j2
                elif (j1==j2):
                    lambda1 = gamma1
                    lambda2 = gamma2
                phase = exp((2*pi * I)/d* (p2*j2 - p1*j2))
                # phase = exp((2*pi * I)/d* (p2*gamma2 - p1*gamma1))
                term = phase*Dagger(var_dict['v'+str(lambda1)])*var_dict['v'+str(lambda2)]
                eqn_ls.append(term)

    eqn= sum(eqn_ls)
    def print_info(): # function to print out individual equations
        print(cp1, cp2)
        print(latex(eqn))
        # print(eqn)
        print('------')
    if pcondition:
        print_info()
    return eqn

## testing sample states ##
# eqn = measure_pair((0,0), (0,1), True)
eqn = measure_pair((1,0), (1,1), True)
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

## determining unique systems ##
def build_map(master_sys):
    def convert_kcp(ktuple): # takes in k-tuple
        vals=[]
        for pair in ktuple:
            c,p = pair[0],pair[1]
            vals.append(c*d +p)
        return tuple(vals)
    
    unique_sys=[] # list to hold all systems that are unique
    checked_sys = [] # list of all matched coordinates of wholly similar systems
    lonely_sys = [] # list to hold all the positions of all systems that share no eqns
    sys_dict = dict() # of the systems that do share equations, log the other positions of the equations and the number they share

    for i, sys in enumerate(master_sys):
        shared_sys = [] # stores list of value coordinates and number of equations shared
        vals = convert_kcp(k_cp[i])
        num_total_shared = 0 # keep track of how many other systems in total this system matched >1 eqn to
        
        # print(k_cp[i])
        # print('-----')
 
        for j, sys1 in enumerate(master_sys):
            if i != j:
                num_shared= 0

                for eqn in sys:
                    cl = Poly(eqn).coeffs()[0] # get coefficient on first term, divide it out to standardize
                    eqn /= cl

                    for eqn1 in sys1:
                        cl1 = Poly(eqn1).coeffs()[0] # get coefficient on first term, divide it out to standardize
                        eqn1 /= cl1
                        # print('eqn', eqn)
                        # print('eqn1', eqn1)
                        # print('eqn=eqn1', eqn==eqn1)
                        if eqn == eqn1: # compare equations
                            num_shared+=1
                # print(num_shared)
                if num_shared > 0:
                    if num_shared ==len(sys): # if all of them are shared, then systems are equivalent
                        num_total_shared+=1
                        vals1 = convert_kcp(k_cp[j])
                        if not((vals in checked_sys) and (vals1 in checked_sys)):
                            unique_sys.append(sys)
                            checked_sys.append(vals)
                            checked_sys.append(vals1)
                            shared_sys.append(vals1)
                    
        if len(shared_sys) > 0:
            sys_dict[vals] = shared_sys
        else: # lonely system!
            lonely_sys.append(vals)
            unique_sys.append(sys)

    print(len(unique_sys), unique_sys)
    print(sys_dict)
    print('lonely', lonely_sys)


    ## make plot ##
    # ax =plt.figure().add_subplot(projection='3d')
    # # ax.scatter(*zip(*lonely_sys),  marker="*")
    # for lonely in lonely_sys:
    #     ax.scatter(list(zip(*lonely))[0], list(zip(*lonely))[1], list(zip(*lonely))[2], marker="*")
    # sys_keys = sys_dict.keys()
    # line_color_ls = ['blue', 'indigo', 'violet']
    # for key in sys_keys:
    #     s_sys = sys_dict[key]
    #     ax.scatter(key[0], key[1], key[2]) # color these based on number of shared
    #     for elems in s_sys:
    #         # get as separate lists
    #         x, y, z = [key[0], elems[0][0]], [key[1], elems[0][1]], [key[2], elems[0][2]]
    #         ax.plot(x,y,z, color=line_color_ls[elems[1]])

    # ax.plt.show()

master_sys = measure_group(k_cp, False, True)
print('-----------')
# print(master_sys)
master_sys_unique = [eqn[0] for eqn in master_sys]
print(master_sys_unique)
# master_sys_unique = list(set(master_sys))
for sys in master_sys_unique:
     print(latex(sys))
# build_map(master_sys)
# print('master sys', len(master_sys))
# for sys in master_sys:
#     print('-----')
#     print(sys)

# measure_pair((0,0), (1,1), pcondition=True)


# old:
#    if sys[l]==sys1[l1]:
#                             num_shared+=1
#                             # print('same')
#                         else:
#                             for m in range(d):
#                                 print('actual', sys[l])
#                                 print('-------')
#                                 print('modified', exp(2*pi * I * m / d)*sys1[l])
#                                 print('-------')
#                                 if sys[l]==exp(2*pi * I * m / d)*sys1[l]:
#                                     num_shared+=1
#                                     print('same')
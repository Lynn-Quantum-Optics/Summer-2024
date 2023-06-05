# file to use in testing mex.py

# using sympy to handle symbolic manipulation
from sympy import *
# from sympy.solvers import solve, solveset
from sympy.physics.quantum.dagger import Dagger
# from sympy.printing.mathematica import MCodePrinter
import numpy as np
from itertools import *
import pandas as pd
import time
import signal # limit time for solvers

## Initialize d value ##

# d = int(input('Enter a value for d: '))
d=2
## Choose k ##
# k = int(input('Enter a value for k: '))
k = 3

## Helper Functions for Bell State Generation ##

# generate bell states in number basis
# e.g. |01> for d = 2: [1, 0, 0, 1]
# key: cp (correlation class, phae class). e.g. 12 => c=1, p=2
bs_dict ={}
def generate_bs(d): 
    # store the phase in separate array to make measurement 
    # adapting equation 2 in the 2019
    for c in range(d): # set correlation class
        for p in range(d): # set phase class
            phase = []# stores phase for each join-particle ket
            # ignoring * 1/np.sqrt(d) factor
            numv_ls = [] # number (state) vector
            for j in range(d): # generate qubit
                phase.append(exp((I*2*pi*p*j)/d))
                numv = np.zeros(2*d)
                numv[j] = 1 #left qudit: from index 0 -> d-1
                numv[(j+c)%d+d] = 1 # right qudit: from index d -> 2d-1
                numv_ls.append(numv) # add the numv to the overall list
            # we've generated a BS at this point
            bs_dict[str(c)+str(p)]=[numv_ls, phase] # store numv and phase in the dict

# function to make BS more readable
def read_BS(bs):
    print_str = '' # initialize total output string
    for i in range(len(bs[0])):
        numv1 = bs[0][i]
        phase1=bs[1][i]
        i1, = np.where(numv1 == 1)
        print_str+=str(phase1)+'|%i%i> + '%(i1[0], i1[1]-d)
    
    print_str = print_str[:-2] # remove extra + at end
    print(print_str)

## Generate Bell States ##
generate_bs(d)
print('*---------*')
print('initializing with d =', d)
print('k = ', k)
print('num BS:', len(bs_dict))
for key in bs_dict:
    read_BS(bs_dict[key])
print('*---------*')

## Helper functions for measurement ##

# initialize symbols
var_dict={}
alphabet = [] # alphabet of a, b synbols
vbet = [] # alphabet of vs
for i in range(2*d):
    # a_i, b_i = Symbol('a'+str(i), real=True), Symbol('b'+str(i), real=True)
    # alphabet['a'+str(i)] = a_i
    # alphabet['b'+str(i)] = b_i
    # alphabet.append(a_i)
    # alphabet.append(b_i)

    # v_i = a_i+b_i*I # split into real and imaginary
    v_i = Symbol('v'+str(i), complex=True)
    var_dict['v'+str(i)] = v_i
    vbet.append(v_i)

# alphabet=tuple(alphabet)
# get normalization term: sum of modulus squared for each variable
norm_ls = [Dagger(var_dict['v'+str(i)])*var_dict['v'+str(i)] for i in range(len(var_dict))]
norm_ls.append(-1) # append a -1, i.e. sum_i v_iv_i^\dagger -1 = 0
norm_cond = sum(norm_ls)

# for measurement
def measure_BS(bs):
    # print('bs', bs)
    # measured_ls = [] # list to hold sympy matrices which we will take the inner product of with other BS
    measured = Matrix(np.zeros(2*d))
    # go through each joint particle state
    for i, numv in enumerate(bs[0]): # 0th entry holds numv_ls, 1st holds phase
        for j in range(2*d): # check over each index to annihilate
            if numv[j]==1: # found a particle, lower it and append lowered numv to measured ls
                lowered = numv.copy() # need to create copy so we don't annihilate on the original, which we will need as we continue iterating
                lowered[j]-=1
                phase = bs[1][i] # get phase factor
                vj = var_dict['v'+str(j)] # get sympy variable coefficient for this annihilation operator
                # break up phase into re and im; need to deal with really small and should be 0 terms


                # phase_re = re(phase)
                # phase_im = im(phase)
                # # print(phase_re, phase_im)
                # if abs(phase_re) < 10**(-4):
                #     phase_re=0
                # if abs(phase_im) < 10**(-4):
                #     phase_im=0
                # print(phase_re, phase_im)
                # phase = nsimplify(phase_re) + nsimplify(phase_im)*I

                coef= phase*vj
                # coef = vj
                # print('coef', coef)
                # if N(coef) < 10**(-4):
                #     coef=0
                result = Matrix(lowered)*coef
                measured+=result
            # do nothing if it's not a 1, since then we delete it
            
    return measured

def print_ex():
    print('-----*----')
    print('measuring:')
    bs1 = bs_dict['01'] # pick sample qudit to measure
    bs2 = bs_dict['10'] # pick sample qudit to measure
    print('--bra-ket--')
    read_BS(bs1)
    read_BS(bs2)
    print()
    print('--number--')
    print(bs1)
    print(bs2)
    print()
    print('---measurement:---')
    mbs1 = measure_BS(bs1)
    mbs2 = measure_BS(bs2)
    print('measured bs1')
    pprint(mbs1)
    print('measured bs2')
    pprint(mbs2)
    print()
    print('----inner product mbs1^\dagger * mbs2:----')
    pprint((Dagger(mbs1)*mbs2)[0])
    # print(latex((Dagger(mbs1)*mbs2)[0]))
    print('-----*----')

print_ex()
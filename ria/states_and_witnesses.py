import numpy as np
import finding_states.operations as op

#######################
## QUANTUM STATES
#######################

# Single-qubit States
H = op.ket([1,0])
V = op.ket([0,1])
R = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
L = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
D = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1)])
A = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1)])

### Jones/Column Vectors & Density Matrices ###
HH = op.ket([1, 0, 0, 0])
HV = op.ket([0, 1, 0, 0])
VH = op.ket([0, 0, 1, 0])
VV = op.ket([0, 0, 0, 1])
HH_RHO = op.get_rho(HH)
HV_RHO = op.get_rho(HV)
VH_RHO = op.get_rho(VH)
VV_RHO = op.get_rho(VV)

### Bell States & Density Matrices ###
PHI_P = (HH + VV)/np.sqrt(2)
PHI_M = (HH - VV)/np.sqrt(2)
PSI_P = (HV + VH)/np.sqrt(2)
PSI_M = (HV - VH)/np.sqrt(2)
PHI_P_RHO = op.get_rho(PHI_P)
PHI_M_RHO = op.get_rho(PHI_M)
PSI_P_RHO = op.get_rho(PSI_P)
PSI_M_RHO = op.get_rho(PSI_M)

### Eritas's States (see spring 2023 writeup) ###
def E0_PSI(eta, chi, rho = False):
    """
    Returns the state of form cos(eta)PSI_P + e^(i*chi)*sin(eta)PSI_M
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = np.cos(eta)*PSI_P + np.exp(chi*1j)*np.sin(eta)*PSI_M

    if rho:
        return op.get_rho(state)
    return state

def E0_PHI(eta, chi, rho = False):
    """
    Returns the state of form cos(eta)PHI_P + e^(i*chi)*sin(eta)PHI_M
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = np.cos(eta)*PHI_P + np.exp(chi*1j)*np.sin(eta)*PHI_M
    if rho:
        return op.get_rho(state)
    return state

def E1(eta, chi, rho = False):
    """
    Returns the state of the form 
    1/sqrt(2) * (cos(eta)*(PSI_P + iPSI_M) + e^(i*chi)*sin(eta)*(PHI_P + iPHI_M))
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = 1/np.sqrt(2) * (np.cos(eta)*(PSI_P + PSI_M*1j) + np.sin(eta)*np.exp(chi*1j)*(PHI_P + PHI_M*1j))
    if rho:
        return op.get_rho(state)
    return state


### Amplitude Damped States ###
def ADS(gamma):
    """
    Returns the amplitude damped state with parameter gamma
    """
    return np.array([[.5, 0, 0, .5*np.sqrt(1-gamma)], 
                     [0, 0, 0, 0], [0, 0, .5*gamma, 0], 
                     [.5*np.sqrt(1-gamma), 0, 0, .5-.5*gamma]])

### Sample state to illustrate power of W5 over W3 ###
def sample_state(phi):
    """
    Returns a state with parameter phi
    """
    ex1 = np.cos(phi)*np.kron(H,D) - np.sin(phi)*np.kron(V,A)
    return op.get_rho(ex1)

##################
## MATRICES
##################

### Pauli Matrices ###
PAULI_X = np.array([[0,1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1,0], [0,-1]])
IDENTITY = np.array([[1,0], [0,1]])

### Rotation Matrices ###
#
# Params: 
#  theta - the angle to rotate by
def R_z(theta):
    return np.array([[np.cos(theta/2) - np.sin(theta/2)*1j, 0], 
                    [0, np.cos(theta/2) + np.sin(theta/2)*1j]])

def R_x(theta):
    return np.array([[np.cos(theta/2), np.sin(theta/2)*1j],
                    [np.sin(theta/2)*1j, np.cos(theta/2)]])

def R_y(theta):
    return np.array([[np.cos(theta/2), -(np.sin(theta/2))],
                    [np.sin(theta/2), (np.cos(theta/2))]])


##########################################
##        ENTANGLEMENT WITNESSES        ## 
##########################################

## Keep track of count indices
# int to str, i.e. COUNT[0] = 'HH'
COUNTS = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 
          'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 
          'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL']
# str to int, i.e. COUNTS_INDEX['HH'] = 0
COUNTS_INDEX = {count: index for index, count in enumerate(COUNTS)} 

class W3:
    """
    W3 (Riccardi) witnesses. These use local measurements on 3 Pauli bases.

    Attributes: 
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data

    NOTE: One of rho or counts must be given
    NOTE: If counts is given, experimental calculations will be used
    NOTE: If rho is given, theoretical calculations will be used
    """
    def __init__(self, rho = None, counts=None):
        self.counts = counts
            
        # counts not given, so we want to use the given theoretical rho
        if not counts:
            assert rho is not None, "ERROR: counts not given, so rho should be given"
            self.rho = rho

        else:
            # counts given, so we will construct an experimental density matrix, so
            # rho should not be given as it's for theoretical states
            assert rho is None, "ERROR: counts was given, so rho should not be given"

            # store individual counts in variables
            self.hh, self.hv = counts[COUNTS_INDEX['HH']], counts[COUNTS_INDEX['HV']]
            self.hd, self.ha = counts[COUNTS_INDEX['HD']], counts[COUNTS_INDEX['HA']]
            self.hr, self.hl = counts[COUNTS_INDEX['HR']], counts[COUNTS_INDEX['HL']]
            self.vh, self.vv = counts[COUNTS_INDEX['VH']], counts[COUNTS_INDEX['VV']]
            self.vd, self.va = counts[COUNTS_INDEX['VD']], counts[COUNTS_INDEX['VA']]
            self.vr, self.vl = counts[COUNTS_INDEX['VR']], counts[COUNTS_INDEX['VL']]
            self.dh, self.dv = counts[COUNTS_INDEX['DH']], counts[COUNTS_INDEX['DV']]
            self.dd, self.da = counts[COUNTS_INDEX['DD']], counts[COUNTS_INDEX['DA']]
            self.dr, self.dl = counts[COUNTS_INDEX['DR']], counts[COUNTS_INDEX['DL']]
            self.ah, self.av = counts[COUNTS_INDEX['AH']], counts[COUNTS_INDEX['AV']]
            self.ad, self.aa = counts[COUNTS_INDEX['AD']], counts[COUNTS_INDEX['AA']]
            self.ar, self.al = counts[COUNTS_INDEX['AR']], counts[COUNTS_INDEX['AL']]
            self.rh, self.rv = counts[COUNTS_INDEX['RH']], counts[COUNTS_INDEX['RV']]
            self.rd, self.ra = counts[COUNTS_INDEX['RD']], counts[COUNTS_INDEX['RA']]
            self.rr, self.rl = counts[COUNTS_INDEX['RR']], counts[COUNTS_INDEX['RL']]
            self.lh, self.lv = counts[COUNTS_INDEX['LH']], counts[COUNTS_INDEX['LV']]
            self.ld, self.la = counts[COUNTS_INDEX['LD']], counts[COUNTS_INDEX['LA']]
            self.lr, self.ll = counts[COUNTS_INDEX['LR']], counts[COUNTS_INDEX['LL']]

            self.stokes = self.calculate_stokes()

            # calculate experimental density matrix
            # NOTE: doesn't necessarily represent a full state
            self.rho = self.expt_rho()


    ##################################
    ## WITNESS DEFINITIONS/FUNCTIONS
    ##################################
    def W3_1(self, theta):
        """
        The first W3 witness

        Params:
        theta - the parameter for the rank-1 projector

        NOTE: this returns the operator, not an expectation value
        """
        a = np.cos(theta)
        b = np.sin(theta)

        # For experimental data, ensure we have the necessary counts
        if self.counts:
            W3.check_counts(self)

        phi1 = a*PHI_P + b*PHI_M
        return op.partial_transpose(phi1 * op.adjoint(phi1))
        
    def W3_2(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)
        
        phi2 = a*PSI_P + b*PSI_M
        return op.partial_transpose(phi2 * op.adjoint(phi2))
        
    def W3_3(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)
        
        phi3 = a*PHI_P + b*PSI_P
        return op.partial_transpose(phi3 * op.adjoint(phi3))

    def W3_4(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)

        phi4 = a*PHI_M + b*PSI_M
        return op.partial_transpose(phi4 * op.adjoint(phi4))
        
    def W3_5(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)

        phi5 = a*PHI_P + 1j*b*PSI_M
        return op.partial_transpose(phi5 * op.adjoint(phi5))
        
    def W3_6(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)

        phi6 = a*PHI_M + 1j*b*PSI_P
        return op.partial_transpose(phi6 * op.adjoint(phi6))

    def get_witnesses(self, vals=False, theta=None):
        """
        Returns either all 6 W3 witnesses as operators or the expectation values 
        of the W3 witnesses with a given theta

        Params:
        vals (optional)  - if true, return the witness expectation values, otherwise return 
                           the operators as functions
        theta (optional) - the theta value to calculate expectation values for when vals is True
        
        NOTE: vals is False by default, so operators are returned by default
        NOTE: theta is not given by default, and must be given when vals is True
        NOTE: operators will always be returned if vals is False even if theta is given
        """
        ws = [self.W3_1, self.W3_2, self.W3_3, self.W3_4, self.W3_5, self.W3_6]
    
        ## By default, return the operators
        if not vals:
            return ws

        ## Otherwise, return the expectation values with the given theta
        assert theta is not None, "ERROR: theta not given"
        ws = [w(theta) for w in ws]
        values = []
        
        for w in ws:
            values += [np.trace(w @ self.rho).real]
        return values


    ##################################
    ## COUNT HANDLING FUNCTIONS
    ##################################
    def expt_rho(self):
        """
        Calculates the experimental density matrix
        
        NOTE: this only represents an actual state if all counts were given
              otherwise, the resulting matrix only contains partial information
              of the state
        """
        
        # Get tensor'd Pauli matrices
        pauli = [np.kron(IDENTITY, IDENTITY), np.kron(IDENTITY, PAULI_X), np.kron(IDENTITY, PAULI_Y),
                 np.kron(IDENTITY, PAULI_Z), np.kron(PAULI_X, IDENTITY), np.kron(PAULI_X, PAULI_X),
                 np.kron(PAULI_X, PAULI_Y), np.kron(PAULI_X, PAULI_Z), np.kron(PAULI_Y, IDENTITY),
                 np.kron(PAULI_Y, PAULI_X), np.kron(PAULI_Y, PAULI_Y), np.kron(PAULI_Y, PAULI_Z), 
                 np.kron(PAULI_Z, IDENTITY), np.kron(PAULI_Z, PAULI_X), np.kron(PAULI_Z, PAULI_Y), 
                 np.kron(PAULI_Z, PAULI_Z)
                ]
        
        rho = np.zeros((4, 4), dtype='complex128')
        for i in range(len(pauli)):
            rho += self.stokes[i] * pauli[i]
        
        return 0.25 * rho

    def calculate_stokes(self):
        """
        Calculates Stokes parameters
        
        NOTE: The order of this list is the same as S_{i, j} as listed in the 
              1-photon and 2-photon states from Beili Nora Hu paper (page 9)
        """
        assert self.counts is not None, "ERROR: counts not given"

        stokes = [1,                                                                         # 0
            (self.dd - self.da + self.ad - self.aa)/(self.dd + self.da + self.ad + self.aa),
            (self.rr + self.lr - self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.hh - self.hv - self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.dd + self.da - self.ad - self.aa)/(self.dd + self.da + self.ad + self.da),
            (self.dd - self.da - self.ad + self.aa)/(self.dd + self.da + self.ad + self.aa), # 5
            (self.dr - self.dl - self.ar + self.al)/(self.dr + self.dl + self.ar + self.al),
            (self.dh - self.dv - self.ah + self.av)/(self.dh + self.dv + self.ah + self.av),
            (self.rr - self.lr + self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.rd - self.ra - self.ld + self.la)/(self.rd + self.ra + self.ld + self.la),
            (self.rr - self.rl - self.lr + self.ll)/(self.rr + self.rl + self.lr + self.ll), # 10
            (self.rh - self.rv - self.lh + self.lv)/(self.rh + self.rv + self.lh + self.lv),
            (self.hh + self.hv - self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.hd - self.ha - self.vd + self.va)/(self.hd + self.ha + self.vd + self.va),
            (self.hr - self.hl - self.vr + self.vl)/(self.hr + self.hl + self.vr + self.vl),
            (self.hh - self.hv - self.vh + self.vv)/(self.hh + self.hv + self.vh + self.vv)  # 15
        ]
        
        return stokes
    
    def check_zz(self, quiet=False):
        """
        Checks the necessary counts to determine if the zz measurement was taken

        Params:
        quiet - if true, don't tell the user if the measurement is given
        """
        assert self.hh != 0, "Missing HH measurement"
        assert self.hv != 0, "Missing HV measurement"
        assert self.vh != 0, "Missing VH measurement"
        assert self.vv != 0, "Missing VV measurement"

        if not quiet:
            print("ZZ measurement was taken!")

    def check_xx(self, quiet=False):
        """Determines if the xx measurement was taken"""
        assert self.dd != 0, "Missing DD measurement"
        assert self.da != 0, "Missing DA measurement"
        assert self.ad != 0, "Missing AD measurement"
        assert self.aa != 0, "Missing AA measurement"

        if not quiet:
            print("XX measurement was taken!")

    def check_yy(self, quiet=False):
        """Determines if the yy measurement was taken"""
        assert self.rr != 0, "Missing RR measurement"
        assert self.rl != 0, "Missing RL measurement"
        assert self.lr != 0, "Missing LR measurement"
        assert self.ll != 0, "Missing LL measurement"

        if not quiet:
            print("YY measurement was taken!")

    def check_xy(self, quiet=False):
        """Determines if the xy measurement was taken"""
        assert self.dr != 0, "Missing DR measurement"
        assert self.dl != 0, "Missing DL measurement"
        assert self.ar != 0, "Missing AR measurement"
        assert self.al != 0, "Missing AL measurement"

        if not quiet:
            print("XY measurement was taken!")

    def check_yx(self, quiet=False):
        """Determines if the yx measurement was taken"""
        assert self.rd != 0, "Missing RD measurement"
        assert self.ra != 0, "Missing RA measurement"
        assert self.ld != 0, "Missing LD measurement"
        assert self.la != 0, "Missing LA measurement"

        if not quiet:
            print("YX measurement was taken!")

    def check_zy(self, quiet=False):
        """Determines if the zy measurement was taken"""
        assert self.hr != 0, "Missing HR measurement"
        assert self.hl != 0, "Missing HL measurement"
        assert self.vr != 0, "Missing VR measurement"
        assert self.vl != 0, "Missing VL measurement"

        if not quiet:
            print("ZY measurement was taken!")

    def check_yz(self, quiet=False):
        """Determines if the yz measurement was taken"""
        assert self.rh != 0, "Missing RH measurement"
        assert self.rv != 0, "Missing RV measurement"
        assert self.lh != 0, "Missing LH measurement"
        assert self.lv != 0, "Missing LV measurement"   

        if not quiet:
            print("YZ measurement was taken!")

    def check_xz(self, quiet=False):
        """Determines if the xz measurement was taken"""
        assert self.dh != 0, "Missing DH measurement"
        assert self.dv != 0, "Missing DV measurement"
        assert self.ah != 0, "Missing AH measurement"
        assert self.av != 0, "Missing AV measurement"

        if not quiet:
            print("XZ measurement was taken!")

    def check_zx(self, quiet=False):
        """Determines if the zx measurement was taken"""
        assert self.hd != 0, "Missing HD measurement"
        assert self.ha != 0, "Missing HA measurement"
        assert self.vd != 0, "Missing VD measurement"
        assert self.va != 0, "Missing VA measurement"

        if not quiet:
            print("ZX measurement was taken!")

    def check_all_counts(self):
        """
        Check that all counts were provided, this is the same
        as taking a full tomography
        """
        self.check_zz(quiet=True)
        self.check_yy(quiet=True)
        self.check_xx(quiet=True)
        self.check_xy(quiet=True)
        self.check_yx(quiet=True)
        self.check_zy(quiet=True)
        self.check_yz(quiet=True)
        self.check_xz(quiet=True)
        self.check_zx(quiet=True)

        print("Full tomography taken!")

    def check_counts(self):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data
        """
        self.check_zz(quiet=True)
        self.check_xx(quiet=True)
        self.check_yy(quiet=True)

    def __str__(self):
        return (
            f'Rho: {self.rho}\n'
            f'Counts: {self.counts}\n'
        )

class W5(W3):
    """
    W5 witnesses, calculated with Paco's rotations (section 3.4.2 of Navarro thesis).
    These use local measurements on 5 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W3, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)

    ## Triplet 1: Rotate about z ##
    def W5_1(self, theta, alpha):
        """
        First W5 witness, rotates particle 1 about the z-axis
        from the W3_1 witness

        Params:
        theta - free parameter used in W3_1
        alpha - rotation angle
        """
        # W3 witness to rotate from
        w1 = self.W3_1(theta)
        
        if self.counts:
            W5.check_counts(self, triplet=1)

        # Create appropriate rotation matrix and apply it to the W3 witness
        rotation = np.kron(R_z(alpha), IDENTITY)
        return op.rotate_m(w1, rotation)
    
    def W5_2(self, theta, alpha):
        w2 = self.W3_2(theta)

        if self.counts:
            W5.check_counts(self, triplet=1)

        rotation = np.kron(R_z(alpha), IDENTITY)
        return op.rotate_m(w2, rotation)
    
    def W5_3(self, theta, alpha, beta):
        w3 = self.W3_3(theta)

        if self.counts:
            W5.check_counts(self, triplet=1)

        rotation = np.kron(R_z(alpha), R_z(beta))
        return op.rotate_m(w3, rotation)
    

    ## Triplet 2: Rotate about x ##
    def W5_4(self, theta, alpha):
        w3 = self.W3_3(theta)

        if self.counts:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), IDENTITY)
        return op.rotate_m(w3, rotation)
    
    def W5_5(self, theta, alpha):
        w4 = self.W3_4(theta)

        if self.counts:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), IDENTITY)
        return op.rotate_m(w4, rotation)
    
    def W5_6(self, theta, alpha, beta):
        w1 = self.W3_1(theta)

        if self.counts:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), R_y(beta))
        return op.rotate_m(w1, rotation)
    
    
    ## Triplet 3: Rotate about y ##
    def W5_7(self, theta, alpha):
        w5 = self.W3_5(theta)

        if self.counts:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), IDENTITY)
        return op.rotate_m(w5, rotation)
    
    def W5_8(self, theta, alpha):
        w6 = self.W3_6(theta)

        if self.counts:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), IDENTITY)
        return op.rotate_m(w6, rotation)

    def W5_9(self, theta, alpha, beta):
        w1 = self.W3_1(theta)

        if self.counts:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), R_y(beta))
        return op.rotate_m(w1, rotation)
    

    def get_witnesses(self, vals=False, theta=None, alpha=None, beta=None):
        """
        Returns either all 9 W5 witnesses as operators (default) or the expectation values 
        of the W5 witnesses with given parameters (if vals is set to True)

        NOTE: theta, alpha, beta must be given when vals is True
        NOTE: Works the same as the get_witnesses function from W3, see docstring from that
              function for more details
        """
        w5s = [self.W5_1, self.W5_2, self.W5_3, 
                self.W5_4, self.W5_5, self.W5_6,
                self.W5_7, self.W5_8, self.W5_9]
        
        ## Return operators if vals is False
        if not vals:
            return w5s
        
        # Otherwise, we calculate expectation values, so check to see that
        # all parameters are given
        assert theta is not None, "ERROR: theta not given"
        assert alpha is not None, "ERROR: alpha not given"
        assert beta is not None, "ERROR: beta not given"

        # Get the W5 operators for the given parameters
        for i, W in enumerate(w5s):
            if i == 2 or i == 5 or i == 8:
                w5s[i] = W(theta, alpha, beta)
            else:
                w5s[i] = W(theta, alpha)
        
        # Calculate the expectation values
        values = []
        for w in w5s:
            values += [np.trace(w @ self.rho).real]
        return values
    
    
    def check_counts(self, triplet):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data

        Params:
        triplet - which triplet the witness belongs to

        NOTE: The xx, yy, and zz measurement have already been checked
              by a previous call to W3.check_counts, which happens when
              getting the W3 witness to perform the rotation on to get the W5
        """
        if triplet == 1:
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)

        elif triplet == 2:
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)

        elif triplet == 3:
            self.check_xz(quiet=True)
            self.check_zx(quiet=True)

        else:
            assert False, "Invalid triplet specified"
    

class W8(W5):
    """
    W8 witnesses, calculated with Paco's rotations (section 4.1 of Navarro thesis).
    These use local measurements on 8 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W5, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)


    ######################################################
    ## SET 1 (W8_{1-6}): EXCLUDES XY MEASUREMENT
    ######################################################
    
    ## Triplet 1: Rotate particle 1 on y then x ##
    def W8_1(self, theta, alpha, beta, for_w7=False):
        """
        First W8 witness

        Params:
        theta             - free parameter used in W3
        alpha             - first rotation angle (for W5)
        beta              - second rotation angle (to go from W5 to W8)
        for_w7 (optional) - if true, don't check the counts assuming 
                            we're using experimental data

        NOTE: We may calculate W8 in order to perform another rotation to
              get a W7 witness. In this case, we don't want to check the counts
              because using a W7 requires less photon counts than a W8
        """
        # W5 witness to rotate from
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=1)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_7, rotation)
    
    def W8_2(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=1)
        
        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_8, rotation)
    
    def W8_3(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=1)
        
        rotation = np.kron(R_x(gamma), IDENTITY)
        return op.rotate_m(w5_9, rotation)
    

    ## Triplet 2: Rotate about x, then rotate particle 2 about y ##
    def W8_4(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_4, rotation)
    
    def W8_5(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_5, rotation)
    
    def W8_6(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(gamma))
        return op.rotate_m(w5_6, rotation)
    

    #############################################
    ## SET 2: EXCLUDES YX
    #############################################

    ## Triplet 3: Rotate particle 1 on x then y ##
    def W8_7(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=3)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_4, rotation)
    
    def W8_8(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=3)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_5, rotation)
    
    def W8_9(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=3)

        rotation = np.kron(R_y(gamma), IDENTITY)
        return op.rotate_m(w5_6, rotation)
    

    ## Triplet 4: Rotate about y, then rotate particle 2 about x ##
    def W8_10(self, theta, alpha, beta, for_w7=False):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_7, rotation)
    
    def W8_11(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_8, rotation)
    
    def W8_12(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(gamma))
        return op.rotate_m(w5_9, rotation)
    
    #############################################
    ## SET 3: EXCLUDES XZ
    #############################################

    ## Triplet 5: Rotate particle 1 by z then x ##
    def W8_13(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=5)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_1, rotation)
    
    def W8_14(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=5)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_2, rotation)
    
    def W8_15(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=5)

        rotation = np.kron(R_x(gamma), IDENTITY)
        return op.rotate_m(w5_3, rotation)
    

    ## Triplet 6: Rotate about x, then rotate particle 2 by z ##
    def W8_16(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_4, rotation)
    
    def W8_17(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_5, rotation)
    
    def W8_18(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(gamma))
        return op.rotate_m(w5_6, rotation)
    

    #############################################
    ## SET 4: EXCLUDES ZX
    #############################################

    ## Triplet 7: Rotate particle 1 by x then z ##
    def W8_19(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=7)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_4, rotation)
    
    def W8_20(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=7)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_5, rotation)
    
    def W8_21(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=7)

        rotation = np.kron(R_z(gamma), IDENTITY)
        return op.rotate_m(w5_6, rotation)
    

    ## Triplet 8: Rotate about z, then rotate particle 2 by x ##
    def W8_22(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_1, rotation)
    
    def W8_23(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_2, rotation)
    
    def W8_24(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(gamma))
        return op.rotate_m(w5_3, rotation)
    
    
    #############################################
    ## SET 5: EXCLUDES YZ
    #############################################

    ## Triplet 9: Rotate particle 1 by z then y ##
    def W8_25(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=9)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_1, rotation)
    
    def W8_26(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=9)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_2, rotation)
    
    def W8_27(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=9)

        rotation = np.kron(R_y(gamma), IDENTITY)
        return op.rotate_m(w5_3, rotation)
    

    ## Triplet 10: Rotate about y, then rotate particle 2 by z ##
    def W8_28(self, theta, alpha, beta, for_w7=False):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_7, rotation)
    
    def W8_29(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_8, rotation)
    
    def W8_30(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(gamma))
        return op.rotate_m(w5_9, rotation)
    

    #############################################
    ## SET 6: EXCLUDES ZY
    #############################################

    ## Triplet 11: Rotate particle 1 by y then z ##
    def W8_31(self, theta, alpha, beta, for_w7=False):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=11)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_7, rotation)
    
    def W8_32(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=11)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_8, rotation)
    
    def W8_33(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=11)

        rotation = np.kron(R_z(gamma), IDENTITY)
        return op.rotate_m(w5_9, rotation)
    

    ## Triplet 12: Rotate about z, then rotate particle 2 by y ##
    def W8_34(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_1, rotation)
    
    def W8_35(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_2, rotation)
    
    def W8_36(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(gamma))
        return op.rotate_m(w5_3, rotation)


    def get_witnesses(self, vals=False, theta=None, alpha=None, beta=None, gamma=None):
        """
        Returns either all 36 W8 witnesses as operators or the expectation values 
        of the W8 witnesses with given parameters

        NOTE: theta, alpha, beta, gamma must be given when vals is True
        NOTE: Works the same as the get_witnesses function from W3, see docstring from that
              function for more details
        """
        w8s = [self.W8_1, self.W8_2, self.W8_3, self.W8_4, self.W8_5, self.W8_6,
               self.W8_7, self.W8_8, self.W8_9, self.W8_10, self.W8_11, self.W8_12,
               self.W8_13, self.W8_14, self.W8_15, self.W8_16, self.W8_17, self.W8_18,
               self.W8_19, self.W8_20, self.W8_21, self.W8_22, self.W8_23, self.W8_24,
               self.W8_25, self.W8_26, self.W8_27, self.W8_28, self.W8_29, self.W8_30,
               self.W8_31, self.W8_32, self.W8_33, self.W8_34, self.W8_35, self.W8_36]
        
        ## Return operators
        if not vals:
            return w8s
        
        # Check to see that all parameters are given
        assert theta is not None, "ERROR: theta not given"
        assert alpha is not None, "ERROR: alpha not given"
        assert beta is not None, "ERROR: beta not given"
        assert gamma is not None, "ERROR: gamma not given"

        # Get the W8 operators for the given parameters
        for i, W in enumerate(w8s):
            # every 3rd witness needs gamma (NOTE: i is zero-indexed)
            if i % 3 == 2:
                w8s[i] = W(theta, alpha, beta, gamma)
            else:
                w8s[i] = W(theta, alpha, beta)
        
        # Calculate the expectation values
        values = []
        for w in w8s:
            values += [np.trace(w @ self.rho).real]
        return values
    
    def check_counts(self, triplet):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data

        Params:
        triplet - which triplet the witness belongs to
        """
        ## Triplets 1-2 exclude the xy measurement ##
        if triplet == 1:
            # Rotation from 3rd W5 triplet
            #
            # already considered: xz, zx, xx, yy, zz
            # need to check: zy, yz, yx
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_yx(quiet=True)
            
        elif triplet == 2:
            # Rotation from 2nd W5 triplet
            #
            # already considered: zy, yz, xx, yy, zz
            # need to check: xz, zx, yx
            self.check_xz(quiet=True)
            self.check_zx(quiet=True)
            self.check_yx(quiet=True)


        ## Triplets 3-4 exclude yx ##
        elif triplet == 3:
            # Rotation from 2nd W5 triplet
            # need to check: xz, zx, xy
            self.check_xz(quiet=True)
            self.check_zx(quiet=True)
            self.check_xy(quiet=True)
        
        elif triplet == 4:
            # Rotation from 3rd W5 triplet
            # need to check: zy, yz, xy
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_xy(quiet=True)
        

        ## Triplets 5-6 exclude xz ##
        elif triplet == 5:
            # Rotation from 1st W5 triplet
            #
            # already considered: xy, yx, xx, yy, zz
            # need to check: zy, yz, zx
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_zx(quiet=True)
        
        elif triplet == 6:
            # Rotation from 2nd W5 triplet
            # need to check: xy, yx, zx
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_zx(quiet=True)
        

        ## Triplets 7-8 exclude zx ##
        elif triplet == 7:
            # Rotation from 2nd W5 triplet
            # need to check: xy, yx, xz
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_xz(quiet=True)

        elif triplet == 8:
            # Rotation from 1st W5 triplet
            # need to check: zy, yz, xz
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_xz(quiet=True)
        

        ## Triplets 9-10 exclude yz ##
        elif triplet == 9:
            # Rotation from 1st W5 triplet
            # need to check: zx, xz, zy
            self.check_zx(quiet=True)
            self.check_xz(quiet=True)
            self.check_zy(quiet=True)
        
        elif triplet == 10:
            # Rotation from 3rd W5 triplet
            # need to check: xy, yx, zy
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_zy(quiet=True)
        

        ## Triplets 11-12 exclude zy ##
        elif triplet == 11:
            # Rotation from 3rd W5 triplet
            # need to check: xy, yx, yz
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_yz(quiet=True)
        
        elif triplet == 12:
            # Rotation from 1st W5 triplet
            # need to check: zx, xz, yz
            self.check_zx(quiet=True)
            self.check_xz(quiet=True)
            self.check_yz(quiet=True)

        else:
            assert False, "Invalid triplet specified"


class W7(W8):
    """
    W7 witnesses, calculated with Paco's rotations (section 4.2 of Navarro thesis).
    These use local measurements on 7 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W8, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)

    
    #############################################
    ## SET 1 (W7_{1-12}): EXCLUDES XY & YX
    #############################################

    ## Triplet 1: deletion rotation about y on particle 2 from first W8 triplet ##
    def W7_1(self, theta, alpha, beta):
        """
        First W8 witness

        Params:
        theta - free parameter used in W3
        alpha - first rotation angle (for W5)
        beta  - second rotation angle (for W8)
        """
        # deletion rotation angle (defined as gamma in Paco's thesis)
        # TODO: make sure this is being calculated correctly. Am I extracting coefficients?
        delta = np.arctan(-1/np.tan(alpha))

        # W8 witness to rotate from
        w8 = self.W8_1(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_2(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_2(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_3(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_3(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    

    ## Triplet 2: deletion rotation by x on particle 1 from 2nd W8 triplet ##
    def W7_4(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_4(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_5(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_5(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_6(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_6(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    ## Triplet 3: rotate by x on particle 2 from 3rd W8 triplet ##
    def W7_7(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_7(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_8(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_8(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_9(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_9(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    ## Triplet 4: rotate by y on particle 1 from 4th W8 triplet ##
    def W7_10(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_10(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_11(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_11(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_12(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_12(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=1)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    #####################################
    ## SET 2: EXCLUDES XY & YZ
    #####################################

    ## Triplets 5-6: rotate by y on particle 2 from 1st & 2nd W8 triplets, respectively ##
    def W7_13(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_1(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_14(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_2(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_15(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_3(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_16(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_4(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_17(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_5(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_18(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_6(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    
    ## Triplets 7-8: rotate by y on particle 1 from 9th & 10th W8 triplets ##
    def W7_19(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_25(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_20(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_26(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_21(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_27(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_22(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_28(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_23(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_29(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_24(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_30(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=2)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    
    #####################################
    ## SET 3: EXCLUDES XY & ZX
    #####################################

    ## Triplets 9-10: rotate by x on particle 1 from 1st & 2nd W8 triplets ##
    def W7_25(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_1(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_26(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_2(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_27(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_3(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_28(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_4(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_29(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_5(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_30(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_6(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    

    ## Triplets 11-12: rotate by x on particle 2 from 7th & 8th W8 triplets ##
    def W7_31(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_19(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_32(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_20(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_33(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_21(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_34(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_22(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_35(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_23(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_36(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_24(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=3)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    

    #####################################
    ## SET 4: EXCLUDES XZ & ZX
    #####################################

    ## Triplets 13-14: rotate by x on particle 1 from 5th & 6th W8 triplets ##
    def W7_37(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_13(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_38(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_14(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_39(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_15(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_40(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_16(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_41(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_17(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_42(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_18(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    ## Triplets 15-16: rotate by x on particle 2 from 7th & 8th W8 triplets ##
    # TODO: THIS IS THE SAME AS TRIPLETS 11-12, CHECK IN W PACO TO MAKE SURE THIS IS RIGHT
    def W7_43(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_19(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_44(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_20(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_45(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_21(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_46(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_22(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_47(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_23(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_48(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_24(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=4)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    

    #####################################
    ## SET 5: EXCLUDES XZ & ZY
    #####################################

    ## Triplets 17-18: rotate by z on particle 2 from 5th & 6th W8 triplets ##
    def W7_49(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_13(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_50(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_14(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_51(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_15(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_52(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_16(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_53(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_17(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_54(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_18(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)


    ## Triplets 19-20: rotate by z on particle 1 from 11th & 12th W8 triplets ##
    def W7_55(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_31(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_56(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_32(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_57(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_33(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_58(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_34(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_59(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_35(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_60(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_36(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=5)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    #####################################
    ## SET 6: EXCLUDES XZ & YX
    #####################################

    ## Triplets 21-22: rotate by x on particle 1 from 5th & 6th W8 triplets ##
    # TODO: THIS IS THE SAME AS TRIPLETS 13-14, CHECK IN W PACO TO MAKE SURE THIS IS RIGHT
    def W7_61(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_13(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_62(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_14(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_63(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_15(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_64(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_16(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_65(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_17(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_66(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_18(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(R_x(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    ## Triplets 23-24: rotate by x on particle 2 from 3rd & 4th W8 triplets ##
    def W7_67(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_7(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_68(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_8(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_69(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_9(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_70(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_10(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_71(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_11(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_72(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_12(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=6)

        rotation = np.kron(IDENTITY, R_x(delta))
        return op.rotate_m(w8, rotation)
    

    #####################################
    ## SET 8: EXCLUDES YX & ZY
    #####################################

    ## Triplets 25-26: rotate by y on particle 1 from 3rd & 4th W8 triplets ##
    def W7_73(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_7(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_74(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_8(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_75(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_9(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_76(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_10(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_77(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_11(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_78(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_12(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    ## Triplets 27-28: rotate by y on particle 2 from 11th & 12th W8 triplets ##
    def W7_79(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_31(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_80(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_32(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_81(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_33(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_82(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_34(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_83(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_35(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_84(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_36(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=7)

        rotation = np.kron(IDENTITY, R_y(delta))
        return op.rotate_m(w8, rotation)

    
    #####################################
    ## SET 8: EXCLUDES YZ & ZY
    #####################################

    ## Triplets 29-30: rotate by y on particle 1 from 9th & 10th W8 triplets ##
    # TODO: THIS IS THE SAME AS TRIPLETS 7-8, CHECK IN W PACO TO MAKE SURE THIS IS RIGHT
    def W7_85(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_25(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_86(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_26(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_87(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_27(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_88(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_28(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_89(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_29(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_90(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_30(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_y(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    ## Triplets 31-32: rotate by z on particle 1 from 11th & 12th W8 triplets ##
    # TODO: THIS IS THE SAME AS TRIPLETS 19-20, CHECK IN W PACO TO MAKE SURE THIS IS RIGHT
    def W7_91(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_31(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_92(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_32(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_93(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_33(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_94(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_34(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_95(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_35(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_96(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_36(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=8)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    

    #####################################
    ## SET 9: EXCLUDES YZ & ZX
    #####################################

    ## Triplets 33-34: rotate by z on particle 2 from 9th & 10th W8 triplets ##
    def W7_97(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_25(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_98(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_26(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_99(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_27(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_100(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_28(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_101(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_29(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    
    def W7_102(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_30(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(IDENTITY, R_z(delta))
        return op.rotate_m(w8, rotation)
    

    ## Triplets 35-36: rotate by z on particle 1 from 7th & 8th W8 triplets ##
    def W7_103(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_19(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_104(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_20(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_105(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_21(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_106(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_22(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_107(self, theta, alpha, beta):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_23(theta, alpha, beta, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)
    
    def W7_108(self, theta, alpha, beta, gamma):
        delta = np.arctan(-1/np.tan(alpha))
        w8 = self.W8_24(theta, alpha, beta, gamma, for_w7=True)

        if self.counts:
            W7.check_counts(set=9)

        rotation = np.kron(R_z(delta), IDENTITY)
        return op.rotate_m(w8, rotation)


    def check_counts(self, set):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data

        Params:
        set - which set of 6 the witness belongs to
        """
        # TODO: not implemented yet
        return


    def get_witnesses(self, vals=False, theta=None, alpha=None, beta=None, gamma=None):
        """
        Returns either all 108 W7 witnesses as operators or the expectation values 
        of the W7 witnesses with given parameters

        NOTE: theta, alpha, beta, gamma must be given when vals is True
        NOTE: Works the same as the get_witnesses function from W3, see docstring from that
              function for more details
        """
        w7s = [self.W7_1, self.W7_2, self.W7_3, self.W7_4, self.W7_5, self.W7_6,
               self.W7_7, self.W7_8, self.W7_9, self.W7_10, self.W7_11, self.W7_12,
               self.W7_13, self.W7_14, self.W7_15, self.W7_16, self.W7_17, self.W7_18,
               self.W7_19, self.W7_20, self.W7_21, self.W7_22, self.W7_23, self.W7_24,
               self.W7_25, self.W7_26, self.W7_27, self.W7_28, self.W7_29, self.W7_30,
               self.W7_31, self.W7_32, self.W7_33, self.W7_34, self.W7_35, self.W7_36,
               self.W7_37, self.W7_38, self.W7_39, self.W7_40, self.W7_41, self.W7_42,
               self.W7_43, self.W7_44, self.W7_45, self.W7_46, self.W7_47, self.W7_48,
               self.W7_49, self.W7_50, self.W7_51, self.W7_52, self.W7_53, self.W7_54,
               self.W7_55, self.W7_56, self.W7_57, self.W7_58, self.W7_59, self.W7_60,
               self.W7_61, self.W7_62, self.W7_63, self.W7_64, self.W7_65, self.W7_66,
               self.W7_67, self.W7_68, self.W7_69, self.W7_70, self.W7_71, self.W7_72,
               self.W7_73, self.W7_74, self.W7_75, self.W7_76, self.W7_77, self.W7_78,
               self.W7_79, self.W7_80, self.W7_81, self.W7_82, self.W7_83, self.W7_84,
               self.W7_85, self.W7_86, self.W7_87, self.W7_88, self.W7_89, self.W7_90,
               self.W7_91, self.W7_92, self.W7_93, self.W7_94, self.W7_95, self.W7_96,
               self.W7_97, self.W7_98, self.W7_99, self.W7_100, self.W7_101, self.W7_102,
               self.W7_103, self.W7_104, self.W7_105, self.W7_106, self.W7_107, self.W7_108]
        
        ## Return operators
        if not vals:
            return w7s
        
        # Check to see that all parameters are given
        assert theta is not None, "ERROR: theta not given"
        assert alpha is not None, "ERROR: alpha not given"
        assert beta is not None, "ERROR: beta not given"
        assert gamma is not None, "ERROR: gamma not given"

        # Get the W8 operators for the given parameters
        for i, W in enumerate(w7s):
            # every 3rd witness needs gamma
            if i % 3 == 2:
                w7s[i] = W(theta, alpha, beta, gamma)
            else:
                w7s[i] = W(theta, alpha, beta)
        
        # Calculate the expectation values
        values = []
        for w in w7s:
            values += [np.trace(w @ self.rho).real]
        return values
    


class NavarroWitness(W7):
    """
    All witnesses defined in Paco's thesis (i.e. W3, W5, W7, and W8)

    This class doesn't define any new witnesses, but just overrides
    the get_witness function to include all witnesses

    NOTE: all functions defined in W3, W5, W7, and W8 work here
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)

    def get_witnesses(self, vals=False, theta=None, alpha=None, beta=None, gamma=None):
        """
        Returns either all witnesses as operators or the expectation values 
        of the witnesses with given parameters

        NOTE: theta, alpha, beta, gamma must be given when vals is True
        NOTE: Works the same as the get_witnesses function from W3, see docstring from that
              function for more details
        """

        # TODO: Implementation of W7s needs to be finished before this works
        return (W3.get_witnesses(self, vals, theta) + 
                W5.get_witnesses(self, vals, theta, alpha, beta) +
                W7.get_witnesses(self, vals, theta, alpha, beta, gamma) + 
                W8.get_witnesses(self, vals, theta, alpha, beta, gamma))


if __name__ == '__main__':
    print("States, Matrices, and Witnesses Loaded.")
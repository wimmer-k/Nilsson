#!/usr/bin/env python
import sys, getopt
import math
import numpy as np
from memoize import Memoize

#verbose = True
verbose = False


def main(argv):

    np.set_printoptions(linewidth=200)
    
    Nmax = 10
    Omega = 3.5
    Parity = 1

    space = (Nmax, Omega, Parity)
    ind, invind = createstates(*space)
    
    par ={'kappa': 0.06, 'mu': 0.5, 'delta': 0.0}
    
    
    hamiltonian(space, invind, par)

def hamiltonian(space, index, pars):
    
    # basis is N,l,ml,ms
    # states input vector of the states
    states = []
    for s in index:
        states.append(index[s])
    # convert to tuple
    states = tuple(states)
    
    nstates = len(states)

    if verbose:
        print states
        print nstates

    nmax,omega,parity = space
    kappa = pars['kappa']
    mu =    pars['mu']
    delta = pars['delta']

    omega_d = pow(1 - 4./3*delta**2 -16./27*delta**3 , -1./6)
    
    #if verbose:
    print "delta = %.2f, kappa = %.2f, mu = %.2f, omega = %.2f" %(delta,kappa,mu,omega_d)
        
    # Hamiltonian matrix
    H = np.zeros(shape=(nstates,nstates))
    for i in range(nstates):
        N,l,ml,ms = states[i]
        print N,l,ml,ms
        # diagonal 
        #        H_0                  # l^2     # <l^2>         # spin orbit            (<l^2> Nilsson/Ragnarson Excersise 6.7, )
        H[i,i] = N + 3./2 - kappa*mu*(l*(l+1) - 1./2*N*(N+3)) - kappa*2*ml*ms
        # using 3/2 and 1/2 makes the code just a touch slower, but the symbolical notation is preferred
        # off diagonal
        for j in range(i+1,nstates):
            N2,l2,ml2,ms2 = states[j]
            if N!=N2:
                continue
            if l!=l2:
                continue
            for sign in [-1.,+1.]:
                if ml2 == ml+sign and ms2 == -sign*1/2 and ms == sign*1/2:
                    # spin orbit-part
                    H[i,j] = -2*kappa* 1/2*np.sqrt( (l-sign*ml)*(l+sign*ml+1) )
            H[j,i] = H[i,j]
    print H

def rhoY20(N,l,ml,ms,N2,l2,ml2,ms):
    # selection rules
    if ms!=ms2 or ml!=ml2:
        return 0
    if not (N==N2 or abs(N-N2)==2):
        return 0
    if not (l==l2 or abs(l-l2)==2):
        return 0
    
@Memoize
def Y20(l,ml,l2,ml2):
    # <l'm'|Y20|l,m>
    np.sqrt( (5./4/np.pi) * (2*l+1)/(2*l2+1) ) * clebsch
    
def createstates(n_max,omega,parity):
    index = {}
    ctr = 0
    # 0,2,4... for positive parity, 1,3,5,... for negative
    for n in range(parity,n_max+1,2):
        for l in range(parity,n+1,2): #checks for parity
            for ml in range(-l,l+1): 
                for ms in [-0.5,0.5]: #ms
                    if omega==ml+ms:
                        index[(n,l,ml,ms)] = ctr
                        ctr = ctr+1
    invindex = {value: key for key, value in index.items()}
    return index, invindex
    
    
if __name__ == "__main__":
   main(sys.argv[1:])

    

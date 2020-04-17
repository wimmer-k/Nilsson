#!/usr/bin/env python
import sys, getopt
import math
import numpy as np
from memoize import Memoize
from wigner import CG
import matplotlib.pyplot as plt

#verbose = True
verbose = False


def main(argv):

    np.set_printoptions(linewidth=300)
    ## simple test
    #test()
    Nmax = int(argv[0])
    Nnuc = (Nmax+1)*(Nmax+2)*(Nmax+3)/3
    Nlev = Nnuc/2
    
    par ={'kappa': 0.06, 'mu': 0.5, 'delta': 0.2}
    diaopt = {'lowd': -0.3, 'highd': 0.3, 'Nd': 20}
    Nd = diaopt['Nd']
    sw = (diaopt['highd'] - diaopt['lowd'] )/(diaopt['Nd']-1)
    deltas = np.arange(-0.3,0.3+sw,sw)
    el = np.zeros((Nlev,Nd))
    #print el
    cur = 0
    for Omega in np.arange(0.5,Nmax+1,1):
        if verbose:
            print "Omega ", Omega
        for Parity in [0,1]:
            if verbose:
                print "Parity ", Parity
            for d in range(Nd):
                delta = deltas[d]
                #print d, delta
                par['delta'] = delta
                r = runnilsson(Nmax,Omega,Parity,par)
                val = r['eValues']
                #print val
                if len(val) == 0:                   
                    continue
                for v in range(len(val)):
                    el[cur+v][d] = val[v]
                #print el
            if len(val) > 0:
                cur = cur + len(val)
    for l in range(Nlev):
        plt.plot(deltas,el[l])

    plt.show()

## calculate the eigenvalues and vectors for a set of parameters
def runnilsson(Nmax, Omega, Parity, pars):
    space = (Nmax, Omega, Parity)
    ind = createstates(*space)
    h = hamiltonian(space, ind, pars)
    val, vec = diagonalize(h)
    return {'pars': pars, 'nstates': len(ind), 'eValues': val, 'eVectors': vec}
    

def test():
    Nmax = 10
    Omega = 3.5
    Parity = 1

    space = (Nmax, Omega, Parity)
    ind = createstates(*space)
    
    par ={'kappa': 0.06, 'mu': 0.5, 'delta': 0.2}
    h = hamiltonian(space, ind, par)
    val, vec = diagonalize(h)
    print val
    
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

    # for volume conservation wx*wy*wz = w^3
    omega_d = pow(1 - 4./3*delta**2 -16./27*delta**3 , -1./6)
    
    if verbose:
        print "delta = %.2f, kappa = %.2f, mu = %.2f, omega = %.2f" %(delta,kappa,mu,omega_d)
        
    # Hamiltonian matrix
    H = np.zeros(shape=(nstates,nstates))
    for i in range(nstates):
        N,l,ml,ms = states[i]
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

    Hd = hamiltonian_delta(states)
    if verbose:
        print -delta*omega_d*4./3*np.sqrt(np.pi/5)
        print Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5))
        print H + Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5))
        
    return H + Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5))

@Memoize
def hamiltonian_delta(states):
    nstates = len(states)
    H = np.zeros(shape=(nstates,nstates))
    for i in range(nstates):
        N,l,ml,ms = states[i]
        for j in range(i,nstates):
            N2,l2,ml2,ms2 = states[j]
            H[i,j] = rhoY20(N,l,ml,ms,N2,l2,ml2,ms2)
            H[j,i] = H[i,j]
    return H
    
@Memoize
def rhoY20(N,l,ml,ms,N2,l2,ml2,ms2):
    # selection rules
    if ms!=ms2 or ml!=ml2:
        return 0
    if not (N==N2 or abs(N-N2)==2):
        return 0
    if not (l==l2 or abs(l-l2)==2):
        return 0
    return r2(N,l,N2,l2) * Y20(l,ml,l2,ml2)
    
@Memoize
def Y20(l,ml,l2,ml2):
    # <l'm'|Y20|l,m>
    return np.sqrt( (5./4/np.pi) * (2*l+1.)/(2*l2+1) ) * CG(l,ml,2,0, l2, ml2) * CG(l,0,2,0,l2,0) 

@Memoize
def r2(N,l,N2,l2):
    # <N',l'|r^2|N,l>
    if N == N2 and l == l2:
        return (N+3./2)
    if N == N2 and l+2 == l2:
        return np.sqrt( (N-l2+2)*(N+l2+1) )
    if N+2 == N2 and l == l2:
        return np.sqrt( (N2-l2)*(N2+l2+1) )*0.5
    if N+2 == N2 and l+2 == l2:
        return np.sqrt( (N2+l2-1)*(N2+l2+1) )*0.5
    if N+2 == N2 and l-2 == l2:
        return np.sqrt( (N2-l2)*(N2-l2-2) )*0.5
    
    
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
    return invindex
    
def diagonalize(ham):
    eValues, eVectors = np.linalg.eig(ham)
    idx = eValues.argsort() 
    eValues = eValues[idx]
    eVectors = eVectors[:,idx]
    if verbose:
        print eValues
        print eVectors
    return eValues, eVectors
        
if __name__ == "__main__":
   main(sys.argv[1:])

    

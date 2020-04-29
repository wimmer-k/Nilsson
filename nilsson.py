#!/usr/bin/env python
import sys, getopt
import math
import numpy as np
from memoize import Memoize
from wigner import CG
import matplotlib.pyplot as plt

#verbose = True
verbose = False
diagram = True # False # 
ell = ["s","p","d","f","g","h","i","j"]

def main(argv):
    global verbose
    parities = ["+","-"]
    np.set_printoptions(linewidth=300)
    ## simple test
    if len(argv)>0 and (argv[0]=="T" or argv[0]=="t"):
        #verbose = True
        test()
        return
    Nmax = int(argv[0])
    if len(argv)>1:
        if argv[1] == "D" or argv[1] == "d":
            diagram = True
        else:
            diagram = False
            pme = int(argv[1])
    Nnuc = (Nmax+1)*(Nmax+2)*(Nmax+3)/3
    Nlev = Nnuc/2
    mu = [0,0,0,0.35,0.45,0.45,0.45,0.40]
    par ={'kappa': 0.05, 'mu': mu[Nmax], 'delta': 0.2}
    diaopt = {'lowd': -0.3, 'highd': 0.3, 'Nd': 31}
    Nd = diaopt['Nd']
    deltas = np.linspace(diaopt['lowd'],diaopt['highd'],num=diaopt['Nd'],endpoint=True)
    el = np.zeros((Nlev,Nd))
    wf = np.zeros((Nlev,Nlev,Nd))
    Nlabel = ["" for r in range(Nlev)]
    Slabel = ["" for r in range(Nlev)]
    QN = [() for r in range(Nlev)]
    cur = 0
    
    for Omega in np.arange(0.5,Nmax+1,1):
        if verbose:
            print "Omega ", Omega
        for Parity in [0,1]:
            if verbose:
                print "Parity ", Parity
            for d in range(len(deltas)):
                delta = deltas[d]
                #print d, delta
                par['delta'] = delta
                r = runnilsson(Nmax,Omega,Parity,par)
                val = r['eValues']
                if r['nstates'] == 0:                   
                    continue
                for v in range(r['nstates']):
                    el[cur+v][d] = val[v]
                    #print r['eVectors'][v]
                    for w in range(len(r['eVectors'][v])):
                        #print r['eVectors'][v][w]
                        wf[cur+v,cur+w,d] = r['eVectors'][v][w]
            #print wf    
            print "---------------------------"
            ctr = [0 for c in range(Nmax+1)]
            for v in range(r['nstates']):
                N = r['states'][v][0]
                nz = N-ctr[N]-(Omega-0.5)
                l = r['states'][v][1]
                n = (N-l)/2 + 1
                j = r['states'][v][1] + r['states'][v][3]
                Lambda = -1
                if (Omega - 0.5 +nz)%2 == N%2:
                    Lambda = Omega - 0.5
                if (Omega + 0.5 +nz)%2 == N%2:
                    Lambda = Omega + 0.5
                Nlabel[cur+v] = "%d/2^{%s}[%d,%d,%d]" % (Omega*2,parities[Parity],N,nz,Lambda)
                Slabel[cur+v] = "%d%s%d/2^{%s}" % (n,ell[l],2*j,parities[Parity])
                QN[cur+v] = (Omega,Parity,N,nz,Lambda)
                print "N = ", N ,", l =",  l ,", ml = ",  r['states'][v][2] ,", ms = ", r['states'][v][3] ,", v = ", v , ", nz = ", nz, ", Lamba = ", Lambda, ", Omega = ", Omega
                ctr[N] = ctr[N] +1
            if r['nstates'] > 0:
                cur = cur + r['nstates']
    #print wf
    if diagram:
        for l in range(Nlev):
            plt.plot(deltas,el[l])
            plt.text(deltas[-1]+0.02,el[l][-1],"$%s$, %d" % (Nlabel[l],l), ha = 'left')
            plt.text(deltas[0]-0.02,el[l][0],"$%s$, %d" % (Nlabel[l],l), ha = 'right')
        plt.xlabel('$\delta$')
        plt.ylabel('$E/\hbar\omega$')
        plt.tick_params(axis='both', which='major')
        plt.tick_params(axis='both', which='minor')
    else:
        for d in [-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30]:
            i = np.where(abs(deltas - d)<0.01)[0]
            print el[pme][i],
        print '\n'
        for l in range(Nlev):
            plt.plot(deltas,wf[pme][l])
            plt.text(deltas[-1]+0.02,wf[pme][l][-1],"$%s$, %d" % (Slabel[l],l), ha = 'left')
            plt.text(deltas[0]-0.02,wf[pme][l][0],"$%s$, %d" % (Slabel[l],l), ha = 'right')
    plt.xlim(deltas[0]-0.2, deltas[-1]+0.2)
    plt.tight_layout()
    plt.show()

## calculate the eigenvalues and vectors for a set of parameters
def runnilsson(Nmax, Omega, Parity, pars):
    space = (Nmax, Omega, Parity)
    states = createstates(*space)
    h = hamiltonian(space, states, pars)
    val, vec = diagonalize(h)
    return {'pars': pars, 'states': states, 'nstates': len(states), 'eValues': val, 'eVectors': vec}

def test():
    Nmax = 1
    Omega = 0.5
    Parity = 1

    space = (Nmax, Omega, Parity)
    ind = createstates(*space)
    print ind
    
    #par ={'kappa': 0.05, 'mu': 0.0, 'delta': 0.02}
    #h = hamiltonian(space, ind, par)
    #val, vec = diagonalize(h)
    #print val
    print "------------------- spherical -------------------"
    par ={'kappa': 0.05, 'mu': 0.0, 'delta': 0.00}
    h = hamiltonian(space, ind, par)
    val, vec = diagonalize(h)
    for v in range(len(val)):
        print "eigenvalue"
        print val[v]
        print "eigenvector"
        print vec[v]
    mm = np.zeros((2,2))
    mm[:,0] = vec[0]
    mm[:,1] = vec[1]
    print mm
    print np.linalg.inv(mm)
    print "basis trafo "
    print np.dot(np.linalg.inv(mm),vec[0])
    print np.dot(np.linalg.inv(mm),vec[1])
    print "------------------- deformation -------------------"
    par ={'kappa': 0.05, 'mu': 0.0, 'delta': 0.20}
    h = hamiltonian(space, ind, par)
    val, vec = diagonalize(h)
    for v in range(len(val)):
        print "eigenvalue"
        print val[v]
        print "eigenvector"
        print vec[v]
    
    print "basis trafo "
    print val[0], np.dot(np.linalg.inv(mm),vec[0])
    print val[1], np.dot(np.linalg.inv(mm),vec[1])
    
#    nz = [-1 for r in range(len(ind))]
#    Lambda = [-1 for r in range(len(ind))]
#    ctr = [0 for r in range(len(ind))]
#    print ctr
#    for v in range(len(ind)):
#        N = ind[v][0]
#        nz[v] = N-ctr[N]
#        print Omega - 0.5, Omega + 0.5
#        if (Omega - 0.5 +nz[v])%2 == N%2:
#            Lambda[v] = Omega - 0.5
#        if (Omega + 0.5 +nz[v])%2 == N%2:
#            Lambda[v] = Omega + 0.5
#        #label[cur+v] = "%d/2^{%s}[%d,%d,%d]" % (Omega*2,parities[Parity],N,nz,Lambda)
#        #QN[cur+v] = (Omega,Parity,N,nz,Lambda, j)
#        print "N = ", N ,", l =",  ind[v][1] ,", ml = ",  ind[v][2] ,", ms = ", ind[v][3] ,", v = ", v, ", nz = ", nz[v], ", Lamba = ", Lambda[v], ", Omega = ", Omega, ", E = ", val[v]
#        ctr[N] = ctr[N] +1
        
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
            if ml+ms != ml2+ms2:
                continue
            #print N,l,ml,ms,"\t",N2,l2,ml2,ms2 
            for sign in [-1.,+1.]:
                if ml2 == ml+sign and ms2 == -sign*1/2 and ms == sign*1/2:
                    # spin orbit-part
                    H[i,j] = -2*kappa* 1/2*np.sqrt( (l-sign*ml)*(l+sign*ml+1) )
            H[j,i] = H[i,j]

    Hd = hamiltonian_delta(states)
    if verbose:
        print "H0 = \n", H
        print "delta(omega) = ",-delta*omega_d*4./3*np.sqrt(np.pi/5)
        print "Hd = \n", Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5))
        print "H = \n", H + Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5))
        
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
    #print index
    #print invindex
    return invindex
    
def diagonalize(ham):
    eValues, eVectors = np.linalg.eig(ham)
    idx = eValues.argsort() 
    eValues = eValues[idx]
    eVectors = eVectors[:,idx]
    if verbose:
        print eValues
        print eVectors
        print np.transpose(eVectors)
    # eigenvectors are columns
    return eValues, np.transpose(eVectors)
        
if __name__ == "__main__":
   main(sys.argv[1:])

    

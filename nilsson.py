#!/usr/bin/env python
import sys, getopt
import math
import numpy as np
from memoize import Memoize
from wigner import CG
import matplotlib.pyplot as plt
#import csv

#verbose = True
verbose = False
diagram = True # False #
ell = ["s","p","d","f","g","h","i","j"]
parities = ["+","-"]
eps = 0.001 # tolerance for energy matching
# todo:
# FIXED - spherical quantum numbers are wrong
# FIXED calculate decoupling parameters, needs j
# - parameters change as function of N, not nexcessarly Nmax, how to deal with this
# - parameters, states and space are returned by runnilsson, but that is not required if the space is part of input
# - commandline interface

def main(argv):
    global verbose
    
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
            plotdecoup = False
            if len(argv)>2 and argv[2] == "A" or argv[2] == "a":
                plotdecoup = True

    ## basic model space and parameters
    Nnuc = (Nmax+1)*(Nmax+2)*(Nmax+3)/3
    Nlev = Nnuc/2
    mu = [0,0,0,0.35,0.45,0.45,0.45,0.40]
    par ={'kappa': 0.05, 'mu': mu[Nmax], 'delta': 0.0}

    #options for the nillson diagram
    diaopt = {'maxd': 0.3, 'Nd': 30}
    Nd = diaopt['Nd']
    deltas = np.linspace(-diaopt['maxd'],diaopt['maxd'],num=diaopt['Nd']*2+1,endpoint=True)

    #storage containers for el = energy levels, wf = wave functions, ap = decoupling parameter
    el = np.zeros((Nlev,Nd*2+1))
    wf = np.zeros((Nlev,Nlev,Nd*2+1))
    ap = np.zeros((Nlev,Nd*2+1))
    # Nillson quantum numbers
    nQN = [() for r in range(Nlev)]
    # quantum numbers of the sperical states (origin)
    sQN = [() for r in range(Nlev)]
    
    cur = 0
    for Omega in np.arange(0.5,Nmax+1,1):
        if verbose:
            print "Omega ", Omega
        for Parity in [0,1]:
            if verbose:
                print "Parity ", Parity

            # calculate at def = 0 for basis transformation
            par['delta'] = 0.0
            #print par
            space = (Nmax, Omega, Parity)
            states = createstates(*space)
            if len(states) < 1:
                continue
            
            # determine basis transformation matrix from spherical calculation
            r = runnilsson(space,states,par)
            spvect = r['eVectors']
            bt = basistrafo(spvect)
            lastwfcoeff = wavefunction(bt,spvect)

            # determine spherical quantum numbers and ordering
            order = []
            slabels = [() for s in range(len(states))]
            for i in range(len(states)):
                N,l,ml,ms = states[i]
                n = (N-l)/2+1
                j = l + ms
                SO = l if ms==+1./2 else -l-1
                E = N+3./2-par['kappa']*par['mu']*(l*(l+1)- 1./2*N*(N+3))-2*par['kappa']*SO*1.0/2
                o = int(np.where(abs(E-r['eValues'])<eps)[0])
                order.append(o)
                slabels[o] = (n,l,j,Parity)
            
            #obalte part
            for d in range(Nd+1):
                delta = deltas[Nd-d]
                par['delta'] = delta
                r = runnilsson(space,states,par)
                val = r['eValues']
                if r['nstates'] == 0:                   
                    continue
                wfcoeff = wavefunction(bt,r['eVectors'])
                
                for v in range(r['nstates']):
                    el[cur+v][Nd-d] = val[v]
                    # check overlap with last calculation, to avoid jumps invert vectors
                    if d > 0 and np.dot(lastwfcoeff[v],wfcoeff[v]) < np.dot(lastwfcoeff[v],-wfcoeff[v]):
                        wfcoeff[v] = -wfcoeff[v]
                    for w in range(len(wfcoeff[v])):
                        ap[cur+v,Nd-d] += pow(-1,slabels[w][2]-0.5)*(slabels[w][2]+0.5)*wfcoeff[v][w]**2
                        wf[cur+v,cur+w,Nd-d] = wfcoeff[v][w]
                lastwfcoeff = wfcoeff
                
            #reset last coefficients
            lastwfcoeff = wavefunction(bt,spvect)
            
            #prolate part
            for d in range(Nd+1):
                delta = deltas[Nd+d]
                par['delta'] = delta
                r = runnilsson(space,states,par)
                val = r['eValues']
                if r['nstates'] == 0:                   
                    continue
                wfcoeff = wavefunction(bt,r['eVectors'])
                
                for v in range(r['nstates']):
                    el[cur+v][Nd+d] = val[v]
                    # check overlap with last calculation, to avoid jumps invert vectors
                    if d > 0 and np.dot(lastwfcoeff[v],wfcoeff[v]) < np.dot(lastwfcoeff[v],-wfcoeff[v]):
                        wfcoeff[v] = -wfcoeff[v]
                    ap[cur+v,Nd+d] = 0
                    for w in range(len(wfcoeff[v])):
                        ap[cur+v,Nd+d] += pow(-1,slabels[w][2]-0.5)*(slabels[w][2]+0.5)*wfcoeff[v][w]**2
                        wf[cur+v,cur+w,Nd+d] = wfcoeff[v][w]
                lastwfcoeff = wfcoeff
            #print wf    
            if verbose:
                print "---------------------------"
            ctr = [0 for c in range(Nmax+1)]
            # asymptitoc quantum numbers            
            for v in range(r['nstates']):
                N = r['states'][v][0]
                nz = N-ctr[N]-(Omega-0.5)
                l = r['states'][v][1]
                Lambda = -1
                if (Omega - 0.5 +nz)%2 == N%2:
                    Lambda = Omega - 0.5
                if (Omega + 0.5 +nz)%2 == N%2:
                    Lambda = Omega + 0.5
                if verbose:
                    print "N = ", N ,", l =",  l ,", ml = ",  r['states'][v][2] ,", ms = ", r['states'][v][3] ,", v = ", v , ", nz = ", nz, ", Lamba = ", Lambda, ", Omega = ", Omega
                nQN[cur+v] = (Omega,Parity,N,nz,Lambda)
                sQN[cur+v] = slabels[v]
                ctr[N] = ctr[N] +1
                
            if r['nstates'] > 0:
                cur = cur + r['nstates']
 
    ## plotting    
    
    axes = []
    plt.title("Nilsson calculation up to $N_{max} = %d$, $\kappa$ = %.2f, $\mu$ = %.2f" % (Nmax,par['kappa'],par['mu']))
    if diagram:
        plt.plot([0,0],[np.min(el),np.max(el)],ls="--",linewidth=1,color='k')
        for l in range(Nlev):
            plt.plot(deltas,el[l], ls="--" if nQN[l][1] == 1 else "-")
            plt.text(deltas[-1]+0.02,el[l][-1],"$%s$, %d" % (Nlabel(nQN[l]),l), ha = 'left')
            plt.text(deltas[0]-0.02,el[l][0],"$%s$, %d" % (Nlabel(nQN[l]),l), ha = 'right')
        plt.ylabel('$E/\hbar\omega$')
    else:
        print "Nilsson level", Nlabel(nQN[pme])
        print "delta "
        for d in [-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30]:
            print d,"\t\t",
        print "\nenergies:"
        for d in [-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30]:
            i = np.where(abs(deltas - d)<eps)[0]
            print el[pme][i],"\t",

        # check if this is K = 1/2
        if nQN[pme][0] == 0.5:
            if plotdecoup:
                axes.append(plt.subplot(2,1,1))
                plt.title("decoupling parameter for $%s$ level" % Nlabel(nQN[pme]))
                plt.plot([deltas[0]-0.2,deltas[-1]+0.2],[0,0],ls="--",linewidth=1,color='k')
                plt.plot([0,0],[np.min(ap[pme]),np.max(ap[pme])],ls="--",linewidth=1,color='k')
                plt.plot(deltas,ap[pme])
                plt.ylabel('$a$')
                axes.append(plt.subplot(2,1,2, sharex = axes[0]))
                # write out to file
                #with open('decop.dat', 'w') as f:
                #    writer = csv.writer(f, delimiter='\t')
                #    writer.writerows(zip(deltas,ap[pme]))
                
            print "\ndecouling parameter"
            for d in [-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30]:
                i = np.where(abs(deltas - d)<eps)[0]
                print ap[pme][i],"\t",
     
        plt.title("wave function composition for $%s$ level" % Nlabel(nQN[pme]))
        print '\nwave funtion composition'
        plt.plot([deltas[0]-0.2,deltas[-1]+0.2],[0,0],ls="--",linewidth=1,color='k')
        plt.plot([0,0],[np.min(wf[pme]),np.max(wf[pme])],ls="--",linewidth=1,color='k')
        for l in range(Nlev):
            if max(wf[pme][l]) > 0 or min(wf[pme][l]) < 0:
                plt.plot(deltas,wf[pme][l])
                plt.text(deltas[-1]+0.02,wf[pme][l][-1],"$%s$" % (Slabel(sQN[l])), ha = 'left')
                plt.text(deltas[0]-0.02,wf[pme][l][0],"$%s$" % (Slabel(sQN[l])), ha = 'right')
                for d in [-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30]:
                    i = np.where(abs(deltas - d)<eps)[0]
                    print wf[pme][l][i],"\t",
                print Slabel(sQN[l])
        plt.ylabel('$C_{\Omega j}$')
                
    plt.xlabel('$\delta$')
    plt.tick_params(axis='both', which='major')
    plt.tick_params(axis='both', which='minor')
    plt.xlim(deltas[0]-0.2, deltas[-1]+0.2)
    plt.tight_layout()
    plt.show()

def Nlabel(QN):
    Omega, Parity, N, nz, Lambda = QN
    return "%d/2^{%s}[%d%d%d]" % (Omega*2,parities[Parity],N,nz,Lambda)

def Slabel(QN):
    n, l, j, Parity = QN
    return "%d%s%d/2^{%s}" % (n,ell[l],2*j,parities[Parity])
            
## calculate the eigenvalues and vectors for a set of parameters, determine model space
def runnilsson(Nmax, Omega, Parity, pars):
    space = (Nmax, Omega, Parity)
    states = createstates(*space)
    h = hamiltonian(space, states, pars)
    val, vec = diagonalize(h)
    return {'pars': pars, 'states': states, 'nstates': len(states), 'eValues': val, 'eVectors': vec}

## calculate the eigenvalues and vectors for a set of parameters, input model space
def runnilsson(space, states, pars):
    h = hamiltonian(space, states, pars)
    val, vec = diagonalize(h)
    return {'pars': pars, 'states': states, 'nstates': len(states), 'eValues': val, 'eVectors': vec}

## determine the wave functions in terms of the spherical states
def wavefunction(trafo, evectors):
    return np.array([np.dot(trafo,v) for v in evectors])
    
## calculate basis transformation
def basistrafo(evectors):
    #print evectors
    mm = np.transpose(evectors)
    if verbose:
        print "matrix of eigenvectors"
        print mm
        print "inverted matrix of eigenvectors"
        print np.linalg.inv(mm)
    return np.linalg.inv(mm)

def test():
    global verbose
    Nmax = 2
    Omega = 0.5
    Parity = 0
    verbose = True
    space = (Nmax, Omega, Parity)
    ind = createstates(*space)
    print ind
    par ={'kappa': 0.05, 'mu': 0.35, 'delta': 0.00}
    h = hamiltonian(space, ind, par)
    val, vec = diagonalize(h)
    bt = basistrafo(vec)
    tt0 = wavefunction(bt,vec)
    print "spherical components"
    print tt0

    print val
    mm = np.transpose(vec)
    C = -2*par['kappa']
    D = -par['kappa']*par['mu']
    for i in range(len(ind)):
        print '========================='
        N,l,ml,ms = ind[i]
        n = (N-l)/2+1
        j = l + ms
        print N,l,ml,ms
        print n,l,j, "%d%s%d/2^{%s}" % (n,ell[l],2*j,parities[Parity])
        SO = l if ms==+1./2 else -l-1
        print SO
        E = N+3./2+D*(l*(l+1)- 1./2*N*(N+3))+C*SO/2
        print E, np.where(abs(E-val)<0.05)[0]
    ##print np.dot(vec, np.dot(h,mm))
    #t =  np.array([np.dot(h,v) for v in vec])
    #tt =  np.transpose(np.array([np.dot(h,v) for v in vec]))
    #print t
    #print tt
    #print vec
    #print t[0]*vec[0]
    #print t[1]*vec[1]
    #print t[2]*vec[2]
    #print t[3]*vec[3]
    #
    #print tt[0]*vec[0]
    #print tt[1]*vec[1]
    #print tt[2]*vec[2]
    #print tt[3]*vec[3]
def testbasis():
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
    print vec
    bt = basistrafo(vec)
    tt0 = wavefunction(bt,vec)
    for v in range(len(val)):
        print "eigenvalue"
        print val[v]
        print "eigenvector"
        print vec[v]
    #mm = np.zeros((2,2))
    #mm[:,0] = vec[0]
    #mm[:,1] = vec[1]
    mm = np.transpose(vec)
    print "matrix of eigenvectors"
    print mm
    print "inverted matrix of eigenvectors"
    print np.linalg.inv(mm)
    print "basis trafo "
    print np.dot(np.linalg.inv(mm),vec[0])
    print np.dot(np.linalg.inv(mm),vec[1])
    print "------------------- deformation -------------------"
    par ={'kappa': 0.05, 'mu': 0.0, 'delta': -0.20}
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
    print "return this"
    print np.array([np.dot(np.linalg.inv(mm),v) for v in vec])

    print "with functions"
    tt = wavefunction(bt,vec)
    print tt0[0]
    print tt[0]
    print -tt[0]

    print np.dot(tt0[0],tt[0])
    print np.dot(tt0[0],-tt[0])

    print np.dot(tt0[1],tt[1])
    print np.dot(tt0[1],-tt[1])

    
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
        print "idx", idx
        print "eValues\n", eValues
        print "eVectors\n", eVectors
        print "eVectors^T\n", np.transpose(eVectors)
    # eigenvectors are columns
    return eValues, np.transpose(eVectors)
        
if __name__ == "__main__":
   main(sys.argv[1:])

    

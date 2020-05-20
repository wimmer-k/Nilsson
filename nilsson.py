#!/usr/bin/env python
import sys, getopt
import math
import numpy as np
from memoize import Memoize
from wigner import CG
import matplotlib.pyplot as plt
#import csv

# debugging and plotting options
verbose = False # Frue #
diagram = True # False #
DeltaN2 = False # True # 

# constants
ell = ["s","p","d","f","g","h","i","j"]
parities = ["+","-"]
eps = 0.001 # tolerance for energy matching

# g factors free proton, neutron
gs = [5.585694, -3.826085]
gl = [1.0     ,  0.0     ]
# quenching for g-factor
quench = 0.9

# rotational g-factor
gR = 16./44

# todo:
# FIXED - spherical quantum numbers are wrong
# FIXED - calculate decoupling parameters, needs j
# FIXED - calculate magnitc moments, g-factors
# - calculate spectroscopic factors
# - g-factor for states with not Omega = K = J
# - parameters change as function of N, not nexcessarly Nmax, how to deal with this
# - parameters, states and space are returned by runnilsson, but that is not required if the space is part of input
# - commandline interface
# - separate pltting function

def main(argv):
    global verbose
    
    np.set_printoptions(linewidth=300)
    ## simple test
    if len(argv)>0 and (argv[0]=="T" or argv[0]=="t"):
        #verbose = True
        test()
        return
    print argv
    Nmax = int(argv[0])
    plotopt = {'diagram': True, 'decoup': False, 'gfact': False, 'sfact': False}
    
    if len(argv)>1:
        if argv[1] == "D" or argv[1] == "d":
            plotopt['diagram'] = True
        else:
            plotopt['diagram'] = False
            pme = int(argv[1])
            if len(argv)>2 and (argv[2] == "A" or argv[2] == "a"):
                plotopt['decoup'] = True
            if len(argv)>2 and (argv[2] == "G" or argv[2] == "g"):
                plotopt['gfact'] = True
            if len(argv)>2 and (argv[2] == "S" or argv[2] == "s"):
                plotopt['sfact'] = True

    ## basic model space and parameters
    Nnuc = (Nmax+1)*(Nmax+2)*(Nmax+3)/3
    Nlev = Nnuc/2
    mu = [0,0,0,0.35,0.45,0.45,0.45,0.40]
    par ={'kappa': 0.05, 'mu': mu[Nmax], 'delta': 0.0}

    #options for the nillson diagram
    diaopt = {'maxd': 0.4, 'Nd': 40}
    Nd = diaopt['Nd']
    deltas = np.linspace(-diaopt['maxd'],diaopt['maxd'],num=diaopt['Nd']*2+1,endpoint=True)

    #storage containers for el = energy levels, wf = wave functions, ap = decoupling parameter
    el = np.zeros((Nlev,Nd*2+1))
    wf = np.zeros((Nlev,Nlev,Nd*2+1))
    ap = np.zeros((Nlev,Nd*2+1))
    a2 = np.zeros((Nlev,Nd*2+1))
    gf = np.zeros((2,Nlev,Nd*2+1))
    # Nillson quantum numbers
    nQN = [() for r in range(Nlev)]
    # quantum numbers of the sperical states (origin)
    sQN = [() for r in range(Nlev)]
    
    cur = 0
    for Omega in np.arange(1./2,Nmax+1,1):
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
            slabels = [() for s in range(len(states))]
            for i in range(len(states)):
                N,l,ml,ms = states[i]
                n = (N-l)/2+1
                j = l + ms
                SO = l if ms==+1./2 else -l-1
                E = N+3./2-par['kappa']*par['mu']*(l*(l+1)- 1./2*N*(N+3))-2*par['kappa']*SO*1.0/2
                o = int(np.where(abs(E-r['eValues'])<eps)[0])
                slabels[o] = (n,l,j,Parity)
            
            #oblate part
            for d in range(Nd+1):
                delta = deltas[Nd-d]
                par['delta'] = delta
                r = runnilsson(space,states,par)
                val = r['eValues']
                vec = r['eVectors']
                if r['nstates'] == 0:                   
                    continue
                
                wfcoeff = wavefunction(bt,vec)
                
                for v in range(r['nstates']):
                    el[cur+v][Nd-d] = val[v]
                    # check overlap with last calculation, to avoid jumps invert vectors
                    if d > 0 and np.dot(lastwfcoeff[v],wfcoeff[v]) < np.dot(lastwfcoeff[v],-wfcoeff[v]):
                        wfcoeff[v] = -wfcoeff[v]
                    # calculate decoupling parameter and wave funtion in (nlj) representation
                    for w in range(len(wfcoeff[v])):
                        ap[cur+v,Nd-d] += pow(-1,slabels[w][2]-1./2)*(slabels[w][2]+1./2)*wfcoeff[v][w]**2
                        wf[cur+v,cur+w,Nd-d] = wfcoeff[v][w]

                    
                    alsum = 0                    
                    for l in range(Parity,Nmax+1,2): #checks for parity
                        for w in range(len(vec[v])):
                            if states[v][0] != states[w][0]:
                                continue
                            if states[w][1] == l:
                                alsum += states[w][3]*vec[v][w]**2 # 3 =  ms
                    for nop in [0,1]: # neutron proton
                        gf[nop,cur+v,Nd-d] = (gs[nop]*quench-gl[nop])*alsum + (gl[nop] - gR)*Omega +gR*(Omega+1)
                            
                    if Omega == 1./2:
                        # calculate decoupling parameter in Nlmlms representation
                        a2[cur+v,Nd-d] = 0
                        al0sum = 0
                        for l in range(Parity,Nmax+1,2): #checks for parity
                            al0 = 0 # amplitudes using Nilsson's notation
                            al1 = 0
                            for w in range(len(vec[v])):
                                if states[v][0] != states[w][0]: # check same N
                                    continue
                                if states[w][1] == l and Omega-states[w][3] == 0:
                                    al0 = vec[v][w]
                                    al0sum += vec[v][w]**2
                                if states[w][1] == l and Omega-states[w][3] == 1:
                                    al1 = vec[v][w]
                            a2[cur+v,Nd-d] += al0**2 + 2*np.sqrt(l*(l+1))*al0*al1
                        a2[cur+v,Nd-d] *= pow(-1,Parity)                        
                        for nop in [0,1]: # neutron proton
                            gf[nop,cur+v,Nd-d] += (gs[nop]*quench-gl[nop])*pow(-1,Omega+1./2+Parity)*1./2*(Omega+1./2)*al0sum**2 + \
                                                 (gl[nop] - gR)*pow(-1,Omega+1./2)*1./2*(Omega+1./2)*a2[cur+v,Nd-d]

                    for nop in [0,1]: # neutron proton
                        gf[nop,cur+v,Nd-d] *= 1./(Omega+1)
                        
                lastwfcoeff = wfcoeff
                
            #reset last coefficients
            lastwfcoeff = wavefunction(bt,spvect)
            
            #prolate part
            for d in range(Nd+1):
                delta = deltas[Nd+d]
                par['delta'] = delta
                r = runnilsson(space,states,par)
                val = r['eValues']
                vec = r['eVectors']
                if r['nstates'] == 0:                   
                    continue
                
                wfcoeff = wavefunction(bt,vec)
                
                for v in range(r['nstates']):
                    el[cur+v][Nd+d] = val[v]
                    # check overlap with last calculation, to avoid jumps invert vectors
                    if d > 0 and np.dot(lastwfcoeff[v],wfcoeff[v]) < np.dot(lastwfcoeff[v],-wfcoeff[v]):
                        wfcoeff[v] = -wfcoeff[v]
                    # calculate decoupling parameter and wave funtion in (nlj) representation
                    ap[cur+v,Nd+d] = 0  # reset / make sure that is empty
                    for w in range(len(wfcoeff[v])):
                        ap[cur+v,Nd+d] += pow(-1,slabels[w][2]-1./2)*(slabels[w][2]+1./2)*wfcoeff[v][w]**2
                        wf[cur+v,cur+w,Nd+d] = wfcoeff[v][w]

                    alsum = 0
                    for l in range(Parity,Nmax+1,2): #checks for parity
                        for w in range(len(vec[v])):
                            if states[v][0] != states[w][0]:
                                continue
                            if states[w][1] == l:
                                alsum += states[w][3]*vec[v][w]**2 # 3 =  ms
                    for nop in [0,1]: # neutron proton
                        gf[nop,cur+v,Nd+d] = (gs[nop]*quench-gl[nop])*alsum + (gl[nop] - gR)*Omega +gR*(Omega+1)
                        
                    if Omega == 1./2:
                        # calculate decoupling parameter in Nlmlms representation
                        a2[cur+v,Nd+d] = 0
                        al0sum = 0
                        for l in range(Parity,Nmax+1,2): #checks for parity
                            al0 = 0 # amplitudes using Nilsson's notation
                            al1 = 0
                            for w in range(len(vec[v])):
                                if states[v][0] != states[w][0]:
                                    continue
                                if states[w][1] == l and Omega-states[w][3] == 0:
                                    al0 = vec[v][w]
                                    al0sum += vec[v][w]**2
                                if states[w][1] == l and Omega-states[w][3] == 1:
                                    al1 = vec[v][w]
                            a2[cur+v,Nd+d] += al0**2 + 2*np.sqrt(l*(l+1))*al0*al1
                        a2[cur+v,Nd+d] *= pow(-1,Parity)
                        for nop in [0,1]: # neutron proton
                            gf[nop,cur+v,Nd+d] += (gs[nop]*quench-gl[nop])*pow(-1,Omega+1./2+Parity)*1./2*(Omega+1./2)*al0sum**2 + \
                                                  (gl[nop] - gR)*pow(-1,Omega+1./2)*1./2*(Omega+1./2)*a2[cur+v,Nd+d]
                            
                    for nop in [0,1]: # neutron proton
                        gf[nop,cur+v,Nd+d] *= 1./(Omega+1)
                        
                lastwfcoeff = wfcoeff
            #print wf    
            if verbose:
                print "---------------------------"
            ctr = [0 for c in range(Nmax+1)]
            # asymptotic quantum numbers            
            for v in range(r['nstates']):
                N = r['states'][v][0]
                nz = N-ctr[N]-(Omega-1./2)
                l = r['states'][v][1]
                Lambda = -1
                if (Omega - 1./2 +nz)%2 == N%2:
                    Lambda = Omega - 1./2
                if (Omega + 1./2 +nz)%2 == N%2:
                    Lambda = Omega + 1./2
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
    if plotopt['diagram']:
        plt.plot([0,0],[np.min(el),np.max(el)],ls="--",linewidth=1,color='k')
        for l in range(Nlev):
            plt.plot(deltas,el[l], ls="--" if nQN[l][1] == 1 else "-")
            plt.text(deltas[-1]+0.02,el[l][-1],"$%s$, %d" % (Nlabel(nQN[l]),l), ha = 'left')
            plt.text(deltas[0]-0.02,el[l][0],"$%s$, %d" % (Nlabel(nQN[l]),l), ha = 'right')
        plt.ylabel('$E/\hbar\omega$')


    elif plotopt['sfact']:
        plt.title("spectroscopic factors for $%s$ level" % Nlabel(nQN[pme]))
        print "delta "
        for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
            print "%.3f\t" % d,
        print '\nspectroscopic factors'
        for l in range(Nlev):
            if max(wf[pme][l]) > 0 or min(wf[pme][l]) < 0:
                sf = np.array([ ( w * CG(0,0, sQN[l][2],nQN[pme][0], sQN[l][2],nQN[pme][0]) )**2 *2  for w in wf[pme][l] ])
                plt.plot(deltas,sf)
                plt.text(deltas[-1]+0.02,sf[-1],"$%s$" % (Slabel(sQN[l])), ha = 'left')
                plt.text(deltas[0]-0.02,sf[0],"$%s$" % (Slabel(sQN[l])), ha = 'right')
                for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                    i = np.where(abs(deltas - d)<eps)[0]
                    print "%.4f\t" % sf[i],
                print Slabel(sQN[l])
        plt.ylabel('$C^2S$')
        
    else:
        print "Nilsson level", Nlabel(nQN[pme])
        print "delta "
        for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
            print "%.3f\t" % d,
        print "\nenergies:"
        for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
            i = np.where(abs(deltas - d)<eps)[0]
            print "%.4f\t" % el[pme][i],

   
        if plotopt['gfact']:
            axes.append(plt.subplot(2,1,1))
            plt.title("g-factor for $%s$ level" % Nlabel(nQN[pme]))
            #plt.plot([deltas[0]-0.2,deltas[-1]+0.2],[0,0],ls="--",linewidth=1,color='k')
            #plt.plot([0,0],[np.min(gf[:][pme]),np.max(gf[:][pme])],ls="--",linewidth=1,color='k')
            plt.plot(deltas,gf[0][pme])
            plt.plot(deltas,gf[1][pme])
            plt.ylabel('$g$')
            axes.append(plt.subplot(2,1,2, sharex = axes[0]))
 

        # check if this is K = 1/2
        if nQN[pme][0] == 1./2:
            if plotopt['decoup']:
                axes.append(plt.subplot(2,1,1))
                plt.title("decoupling parameter for $%s$ level" % Nlabel(nQN[pme]))
                plt.plot([0,0],[np.min(ap[pme]),np.max(ap[pme])],ls="--",linewidth=1,color='k')
                plt.plot(deltas,ap[pme])
                plt.plot(deltas,a2[pme])
                plt.ylabel('$a$')
                axes.append(plt.subplot(2,1,2, sharex = axes[0]))
                ## write out to file
                #with open('decop.dat', 'w') as f:
                #    writer = csv.writer(f, delimiter='\t')
                #    writer.writerows(zip(deltas,ap[pme]))
                
            print "\ndecouling parameter"
            for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                i = np.where(abs(deltas - d)<eps)[0]
                print "%.4f\t" % ap[pme][i],
            
        plt.title("wave function composition for $%s$ level" % Nlabel(nQN[pme]))
        print '\nwave funtion composition'
        plt.plot([deltas[0]-0.2,deltas[-1]+0.2],[0,0],ls="--",linewidth=1,color='k')
        plt.plot([0,0],[np.min(wf[pme]),np.max(wf[pme])],ls="--",linewidth=1,color='k')
        for l in range(Nlev):
            if max(wf[pme][l]) > 0 or min(wf[pme][l]) < 0:
                plt.plot(deltas,wf[pme][l])
                plt.text(deltas[-1]+0.02,wf[pme][l][-1],"$%s$" % (Slabel(sQN[l])), ha = 'left')
                plt.text(deltas[0]-0.02,wf[pme][l][0],"$%s$" % (Slabel(sQN[l])), ha = 'right')
                for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                    i = np.where(abs(deltas - d)<eps)[0]
                    print "%.4f\t" % wf[pme][l][i],
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
    Nmax = 3
    Omega = 3.5
    Parity = 1
    #verbose = True
    space = (Nmax, Omega, Parity)
    states = createstates(*space)
    print states
    par ={'kappa': 0.05, 'mu': 0.35, 'delta': 0.00}
    h = hamiltonian(space, states, par)
    val, vec = diagonalize(h)
    bt = basistrafo(vec)

    # determine spherical quantum numbers and ordering
    slabels = [() for s in range(len(states))]
    for i in range(len(states)):
        N,l,ml,ms = states[i]
        n = (N-l)/2+1
        j = l + ms
        SO = l if ms==+1./2 else -l-1
        E = N+3./2-par['kappa']*par['mu']*(l*(l+1)- 1./2*N*(N+3))-2*par['kappa']*SO*1.0/2
        o = int(np.where(abs(E-val)<eps)[0])
        slabels[o] = (n,l,j,Parity)

    wfcoeff = wavefunction(bt,vec)
    for v in range(len(val)):
        print "eigenvalue"
        print val[v]
        print "eigenvector"
        print vec[v]
    
    print "-------------"
    ap = [0 for _ in range(len(val))]
    ap2 = [0 for _ in range(len(val))]
    for v in range(len(val)):
        for w in range(len(wfcoeff[v])):
            ap[v] += pow(-1,slabels[w][2]-1./2)*(slabels[w][2]+1./2)*wfcoeff[v][w]**2

        for w in range(len(vec[v])):
            print '%.3f\t' %vec[v][w],
        print "\n",

        # g-factor
        alsum = 0
        for l in range(Parity,Nmax+1,2): #checks for parity
            for w in range(len(vec[v])):
                if states[v][0] != states[w][0]:
                    continue
                alsum += states[w][3]*vec[v][w]**2 # 3 =  ms
        print "alsum ",alsum

        # decoupling parameter
        for l in range(Parity,Nmax+1,2): #checks for parity
            al0 = 0
            al1 = 0
            print "l = ", l
            #print "vec = ", vec[v]
            for w in range(len(vec[v])):
                print "States ", states[w], ", Omega-states[w][3] ", Omega-states[w][3]
                if al0 == 0 and  states[w][1] == l and Omega-states[w][3] == 0:
                    al0 = vec[v][w]
                if al1 == 0 and  states[w][1] == l and Omega-states[w][3] == 1:
                    al1 = vec[v][w]
            print "al0 = " ,al0
            print "al1 = " ,al1
            ap2[v] += al0**2 + 2*np.sqrt(l*(l+1))*al0*al1
            
        print states[v], Omega-states[v][3]
        #ap2[v] = pow(-1,Parity)* (vec[v][3]**2 + 2*np.sqrt(2*(2+1))*vec[v][3]*vec[v][2])
        #print "\n",
        
    ap2 *= pow(-1,Parity)    
    print ap
    print ap2
            
    ## test nilsson eVal table 1
    #Neta = -4
    #delta = np.linspace(-0.4,0.4,200)
    #eta = np.array([ d/par['kappa']*(1-4./3*d**2-16./27*d**3) for d in delta] )

    #plt.plot(delta,eta)
    #plt.show()
    #print delta[np.where(abs(eta-Neta)<0.05)]


    ##par ={'kappa': 0.05, 'mu': 0.45, 'delta': -0.2115}
    ###par ={'kappa': 0.05, 'mu': 0.0, 'delta': -0.2115}
    ##h = hamiltonian(space, ind, par)
    ##val, vec = diagonalize(h)
    ##tt0 = wavefunction(bt,vec)
    ##
    ##print "eigen values"
    ##print val
    ##print "eigen vectors"
    ##print vec
    ##print "wave function"
    ##print tt0
    ##
    ##
    ###print "-------------"
    ###print vec[1]/vec[1][0]
    ###print vec[0]/vec[0][0]
    ##print "-------------"
    ##print vec[5]/vec[5][4]
    ##print vec[4]/vec[4][4]
    ##print vec[3]/vec[3][4]
    ##print vec[2]/vec[2][4]

def testQN():
    global verbose
    Nmax = 2
    Omega = 1./2
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
    Omega = 1./2
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
    # option to include the DeltaN=2 admixtures
    if not DeltaN2 and N!=N2: 
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
                for ms in [-1./2,+1./2]: #ms
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
    # eigenvectors are columns, therefore transpose
    return eValues, np.transpose(eVectors)
        
if __name__ == "__main__":
   main(sys.argv[1:])

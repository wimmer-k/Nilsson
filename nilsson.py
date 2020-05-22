#!/usr/bin/env python3
# for consistent print in python 2 and 3
from __future__ import print_function
import sys, getopt
import math
import numpy as np
from memoize import Memoize
from wigner import CG
from hamiltonian import Nilsson
import matplotlib.pyplot as plt
#import csv

# debugging and plotting options
verbose = False # True # 
diagram = True # False #
DeltaN2 = True # False # 

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
# DONE - calculate decoupling parameters, needs j
# DONE - calculate magnitc moments, g-factors
# DONE - calculate spectroscopic factors
# - g-factor for states with not Omega = K = J
# - parameters change as function of N, not nexcessarly Nmax, how to deal with this
# DONE - parameters, states and space are returned by runnilsson, but that is not required if the space is part of input
# DONE - commandline interface
# DONE - test with python 3
# - separate plotting function
# DONE - put calculation into separate class
# - if not the whole diagram is needed calculate only the states of this Omega and parity

def main(argv):
    global verbose

    # defaults
    Nmax = 3
    plotorb = -1
    plotopt = {'diagram': True, 'decoup': False, 'gfact': False, 'sfact': False}
    
    ## read commandline arguments
    try:
        opts, args = getopt.getopt(argv,"hv:N:o:p:t",["Nmax=","orbit=","pplot=","test"])
    except getopt.GetoptError:
        print('usage: nilsson.py -N <N_max> -o <orbital> -p <property to plot>' )
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('usage: nilsson.py -N <N_max> -o <orbital> -p <property to plot>')
            sys.exit()
        elif opt == '-v':
            verbose = TRUE
        elif opt =="-t":
            print("testing calculation")
            test()
            sys.exit()            
        elif opt in ("-N", "--Nmax"):
            Nmax = int(arg)
        elif opt in ("-o", "--orbit"):
            plotopt['diagram'] = False
            plotorb = int(arg)
        elif opt in ("-p", "--pplot"):
            if plotorb == -1:
                print('need number or orbital to be plotted')
                print('usage: nilsson.py -N <N_max> -o <orbital> -p <property to plot>')
                sys.exit() 
            if arg in ("decoup","a","A"):
                plotopt['decoup'] = True
            elif arg in ("gfact","g","G"):
                plotopt['gfact'] = True
            elif arg in ("sfact","s","S"):
                plotopt['sfact'] = True
            else:
                print('plotting options are:')
                print('\"decoup\", \"a\", or \"A\" for the decoupling parameter')
                print('\"gfact\", \"g\", or \"G\" for the g-factor')
                print('\"sfact\", \"s\", or \"S\" for the spectroscopic factors')
                sys.exit()
                
    np.set_printoptions(linewidth=300)


    ## basic model space and parameters
    Nnuc = (Nmax+1)*(Nmax+2)*(Nmax+3)/3
    Nlev = int(Nnuc/2)
    mu = [0,0,0,0.35,0.45,0.45,0.45,0.40]
    par ={'kappa': 0.05, 'mu': mu[Nmax], 'delta': 0.0}

    #options for the nillson diagram
    diaopt = {'maxd': 0.4, 'Nd': 40}
    Nd = int(diaopt['Nd'])
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
            print("Omega ", Omega)
        for Parity in [0,1]:
            if verbose:
                print("Parity ", Parity)

            nilsson = Nilsson(Nmax = Nmax, Omega = Omega, Parity = Parity, Verbose=verbose)
            nilsson.setparameters(*par)
            nstates = len(nilsson.states)
            states = nilsson.states
            if nstates < 1:
                continue
        
            # determine basis transformation matrix from spherical calculation
            spval, spvect = nilsson.calculate(0.0)
            bt = nilsson.basistrafo(spvect)
            lastwfcoeff = nilsson.wavefunction(bt,spvect)

            # determine spherical quantum numbers and ordering
            slabels = [() for s in range(nstates)]
            for i in range(nstates):
                N,l,ml,ms = states[i]
                n = (N-l)/2+1
                j = l + ms
                SO = l if ms==+1./2 else -l-1
                E = N+3./2-par['kappa']*par['mu']*(l*(l+1)- 1./2*N*(N+3))-2*par['kappa']*SO*1.0/2
                o = int(np.where(abs(E-spval)<eps)[0])
                slabels[o] = (n,l,j,Parity)
            
            #oblate part
            for d in range(Nd+1):
                delta = deltas[Nd-d]
                val, vec = nilsson.calculate(delta)
                
                wfcoeff = nilsson.wavefunction(bt,vec)
                
                for v in range(nstates):
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
            lastwfcoeff = nilsson.wavefunction(bt,spvect)
            
            #prolate part
            for d in range(Nd+1):
                delta = deltas[Nd+d]
                val, vec = nilsson.calculate(delta)
                
                wfcoeff = nilsson.wavefunction(bt,vec)
                
                for v in range(nstates):
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
            #print the wave function and quantum numbers   
            if verbose:
                print("---------------------------")
            ctr = [0 for c in range(Nmax+1)]
            # asymptotic quantum numbers            
            for v in range(nstates):
                N = states[v][0]
                nz = N-ctr[N]-(Omega-1./2)
                l = states[v][1]
                Lambda = -1
                if (Omega - 1./2 +nz)%2 == N%2:
                    Lambda = Omega - 1./2
                if (Omega + 1./2 +nz)%2 == N%2:
                    Lambda = Omega + 1./2
                if verbose:
                    print("N = ", N ,", l =",  l ,", ml = ",  states[v][2] ,", ms = ", states[v][3] ,", v = ", v , ", nz = ", nz, ", Lamba = ", Lambda, ", Omega = ", Omega)
                nQN[cur+v] = (Omega,Parity,N,nz,Lambda)
                sQN[cur+v] = slabels[v]
                ctr[N] = ctr[N] +1
                
            if nstates > 0:
                cur = cur + nstates
 
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
        plt.title("spectroscopic factors for $%s$ level" % Nlabel(nQN[plotorb]))
        print("delta ")
        for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
            print("%.3f\t" % d,end="")
        print('\nspectroscopic factors')
        for l in range(Nlev):
            if max(wf[plotorb][l]) > 0 or min(wf[plotorb][l]) < 0:
                sf = np.array([ ( w * CG(0,0, sQN[l][2],nQN[plotorb][0], sQN[l][2],nQN[plotorb][0]) )**2 *2  for w in wf[plotorb][l] ])
                plt.plot(deltas,sf)
                plt.text(deltas[-1]+0.02,sf[-1],"$%s$" % (Slabel(sQN[l])), ha = 'left')
                plt.text(deltas[0]-0.02,sf[0],"$%s$" % (Slabel(sQN[l])), ha = 'right')
                for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                    i = np.where(abs(deltas - d)<eps)[0]
                    print("%.4f\t" % sf[i],end="")
                print(Slabel(sQN[l]))
        plt.ylabel('$C^2S$')
        
    else:
        print("Nilsson level", Nlabel(nQN[plotorb]))
        print("delta ")
        for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
            print("%.3f\t" % d,end="")
        print("\nenergies:")
        for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
            i = np.where(abs(deltas - d)<eps)[0]
            print("%.4f\t" % el[plotorb][i],end="")

   
        if plotopt['gfact']:
            axes.append(plt.subplot(2,1,1))
            plt.title("g-factor for $%s$ level" % Nlabel(nQN[plotorb]))
            plt.plot(deltas,gf[0][plotorb])
            plt.plot(deltas,gf[1][plotorb])
            plt.text(deltas[-1]+0.02,gf[0][plotorb][-1],"proton", ha = 'left')
            plt.text(deltas[0]-0.02,gf[0][plotorb][0],"proton", ha = 'right')
            plt.text(deltas[-1]+0.02,gf[1][plotorb][-1],"neutron", ha = 'left')
            plt.text(deltas[0]-0.02,gf[1][plotorb][0],"neutron", ha = 'right')
            plt.ylabel('$g$')
            axes.append(plt.subplot(2,1,2, sharex = axes[0]))
            print("\ng-factor parameter")
            for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                i = np.where(abs(deltas - d)<eps)[0]
                print("%.4f\t" % gf[0][plotorb][i],end="")
            print("proton")
            for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                i = np.where(abs(deltas - d)<eps)[0]
                print("%.4f\t" % gf[1][plotorb][i],end="")
            print("neutron")

        # check if this is K = 1/2
        if nQN[plotorb][0] == 1./2:
            if plotopt['decoup']:
                axes.append(plt.subplot(2,1,1))
                plt.title("decoupling parameter for $%s$ level" % Nlabel(nQN[plotorb]))
                plt.plot([0,0],[np.min(ap[plotorb]),np.max(ap[plotorb])],ls="--",linewidth=1,color='k')
                plt.plot(deltas,ap[plotorb])
                plt.plot(deltas,a2[plotorb])
                plt.ylabel('$a$')
                axes.append(plt.subplot(2,1,2, sharex = axes[0]))
                ## write out to file
                #with open('decop.dat', 'w') as f:
                #    writer = csv.writer(f, delimiter='\t')
                #    writer.writerows(zip(deltas,ap[plotorb]))
                
            print("decouling parameter")
            for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                i = np.where(abs(deltas - d)<eps)[0]
                print("%.4f\t" % ap[plotorb][i],end="")
            
        plt.title("wave function composition for $%s$ level" % Nlabel(nQN[plotorb]))
        print('\nwave funtion composition')
        plt.plot([deltas[0]-0.2,deltas[-1]+0.2],[0,0],ls="--",linewidth=1,color='k')
        plt.plot([0,0],[np.min(wf[plotorb]),np.max(wf[plotorb])],ls="--",linewidth=1,color='k')
        for l in range(Nlev):
            if max(wf[plotorb][l]) > 0 or min(wf[plotorb][l]) < 0:
                plt.plot(deltas,wf[plotorb][l])
                plt.text(deltas[-1]+0.02,wf[plotorb][l][-1],"$%s$" % (Slabel(sQN[l])), ha = 'left')
                plt.text(deltas[0]-0.02,wf[plotorb][l][0],"$%s$" % (Slabel(sQN[l])), ha = 'right')
                for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                    i = np.where(abs(deltas - d)<eps)[0]
                    print("%.4f\t" % wf[plotorb][l][i],end="")
                print(Slabel(sQN[l]))
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
            

        
if __name__ == "__main__":
   main(sys.argv[1:])

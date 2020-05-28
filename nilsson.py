#!/usr/bin/env python3
# for consistent print in python 2 and 3
from __future__ import print_function
import sys
import math
import numpy as np
from wigner import CG
from hamiltonian import Nilsson
import matplotlib.pyplot as plt
import argparse

version = "0.2/2020.05.22"

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
# DONE option to calculate only a range of N = [Nmin,Nmax]
# - if not the whole diagram is needed calculate only the states of this Omega and parity (for gui the wave function is stored.)
# DONE option to change kappa and mu from standard values.
# - output g_K instead / in addition
# - in the option to calculate a, also output the magnetic decoupling b

def main(argv):
    
    np.set_printoptions(linewidth=300)
    
    # defaults
    verbose = False
    Nmin = 3
    Nmax = 3
    plotorb = -1
    plotopt = {'diagram': True, 'wavef': False, 'decoup': False, 'gfact': False, 'sfact': False}
    ranged = 0.4
    Nd = 40
    kappa = 0.05
    mu = -1
    savefilepath = None
    
    argparser = argparse.ArgumentParser(description='Calculation of Nilsson diagrams and wave functions.', add_help=False)
    requiredargs = argparser.add_argument_group('required arguments')
    requiredargs.add_argument("-N", "-n", "--nosc", required=True,nargs="+",dest="Nosc",type=int,help="oscillator shell N, or a range of oscillator shells Nmin Mmax, e.g. -N 3 or -N 1 3")

    argparser.add_argument("-o", "--orbital", dest="orb", type=int,help="number of orbital to be plotted")
    argparser.add_argument("-p", "--property", dest="prop", choices=['wavef','wf','decoup','a', 'gfactor','gfact','g', 'sfactor','sfact','s'], help="which property should be plotted. The options are wave function: \"wavef\" or \"wf\", decoupling parameter: \"decoup\" or \"a\", g-factor: \"gfactor\", \"gfact\", or \"g\", spectroscopic factors: \"sfactor\", \"sfact\", or \"s\"")
    
    argparser.add_argument("-k","--kappa", dest="kappa", type=float,help="kappa value for the energy calculations, default value is 0.05")
    argparser.add_argument("-m","--mu", dest="mu", type=float,help="mu value for the energy calculations, default value depend on the value of N")
    argparser.add_argument("-Nd","--ndelta", dest="Nd", type=int,help="number of steps in delta, typical value is 40")
    argparser.add_argument("-r","--ranged", dest="ranged", type=float,help="range of delta values [-ranged,ranged], typical value is 0.4")
    argparser.add_argument("-w","--write", dest="write", type=str,help="write the calculation to output file")

    argparser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    argparser.add_argument("-t", "--test", help="excute testing functions", action="store_true")
    argparser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    argparser.add_argument('--version', action='version', version=version)
    

    argparser._action_groups.reverse()
    
    args = argparser.parse_args()

    if args.test:
        print("test calculation")
        test()
        sys.exit()            

    if not (1<=len(args.Nosc)<=2):
        raise argparser.error("-N expects one or two values, oscillator shell N, or range of oscillator shells -N 3 or -N 1 3")
    
    if args.verbose:
        verbose = TRUE
    if len(args.Nosc) == 1:
        Nmin = args.Nosc[0]
        Nmax = args.Nosc[0]
        print("calculation for N = %d" % Nmax)
    if len(args.Nosc) == 2:
        Nmin = min(args.Nosc)
        Nmax = max(args.Nosc)
        print("calculation for N = [%d,%d]" % (Nmin, Nmax))
    if args.orb is not None and args.orb > -1:
        plotopt['diagram'] = False
        plotorb = args.orb
        if args.prop:
            if args.prop in ("wf","wf"):
                plotopt['wavef'] = True
            if args.prop in ("decoup","a"):
                plotopt['decoup'] = True
            if args.prop in ("gfactor","gfact","g"):
                plotopt['gfact'] = True
            if args.prop in ("sfactor","sfact","s"):
                plotopt['sfact'] = True
    elif args.prop is not None:
        raise argparser.error("-p/--property needs to specify which orbit, e.g. \" -o 3 \"")

    if args.Nd is not None:
        if args.Nd < 1:
            raise argparser.error("-Nd/--ndelta must be 1 or larger, typically around 40")
        else:
            Nd = args.Nd
    if args.ranged is not None:
        if args.ranged < 0:
            raise argparser.error("-r/--ranged must be positive, typically around 0.4")
        else:
            ranged = args.ranged
    if args.kappa is not None:
        if args.kappa < 0:
            raise argparser.error("-k/--kappa must be positive, default value is 0.05")
        else:
            kappa = args.kappa
    if args.mu is not None:
        if args.mu < 0:
            raise argparser.error("-m/--mu must be 0 or positive, default value is between 0 and 0.45 depending on the shell")
        else:
            mu = args.mu

    if args.write is not None:
        savefilepath = args.write
    
    
    print("%d steps in delta in [%.2f,%.2f]"%(Nd,ranged,ranged))
    


    ## basic model space and parameters
    Nnuc = (Nmax+1)*(Nmax+2)*(Nmax+3)/3 - (Nmin-1+1)*(Nmin-1+2)*(Nmin-1+3)/3
    Nlev = int(Nnuc/2)
    print("number of Nilsson levels:", Nlev)
    if plotorb >= Nlev:
        raise argparser.error("invalid orbit selected, number of Nilsson levels = %d, cannot plot orbit orb = %d (0<=orb<N)" % (Nlev,plotorb))
    muN = [0,0,0,0.35,0.45,0.45,0.45,0.40]
    par ={'kappa': kappa, 'mu': muN[Nmax], 'delta': 0.0}
    if mu != -1:
        par['mu'] = mu
    print("kappa = %.3f, mu = %.3f" %(par['kappa'],par['mu']))
    
    #options for the nillson diagram
    diaopt = {'mind': -ranged, 'maxd': ranged, 'Nd': Nd}
    Nd = int(diaopt['Nd'])
    deltas = np.linspace(diaopt['mind'],diaopt['maxd'],num=diaopt['Nd']*2+1,endpoint=True)

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

            nilsson = Nilsson(Nmin = Nmin, Nmax = Nmax, Omega = Omega, Parity = Parity, Verbose=verbose)
            nilsson.setparameters(**par)
            nstates = len(nilsson.states)
            states = nilsson.states
            if nstates < 1:
                continue
            
            """ determine basis transformation matrix from spherical calculation """
            spval, spvect = nilsson.calculate(0.0)
            bt = nilsson.basistrafo(spvect)
            lastwfcoeff = nilsson.wavefunction(bt,spvect)

            """ determine spherical quantum numbers and ordering """
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
    if Nmin == Nmax:
        plt.title("Nilsson calculation for $N = %d$, $\kappa$ = %.2f, $\mu$ = %.2f" % (Nmax,par['kappa'],par['mu']))
    else:
        plt.title("Nilsson calculation $N = [%d,%d]$, $\kappa$ = %.2f, $\mu$ = %.2f" % (Nmin,Nmax,par['kappa'],par['mu']))
    if plotopt['diagram']:
        plt.plot([0,0],[np.min(el),np.max(el)],ls="--",linewidth=1,color='k')
        for l in range(Nlev):
            plt.plot(deltas,el[l], ls="--" if nQN[l][1] == 1 else "-")
            plt.text(deltas[-1]+0.02,el[l][-1],"$%s$, %d" % (Nlabel(nQN[l]),l), ha = 'left')
            plt.text(deltas[0]-0.02,el[l][0],"$%s$, %d" % (Nlabel(nQN[l]),l), ha = 'right')
        plt.ylabel('$E/\hbar\omega$')
        ## write out to file
        if savefilepath is not None:
            with open(savefilepath, "w") as output:
                if Nmax == Nmax:
                    ntext = "N = %d" % Nmax
                else:
                    ntext = "N = [%d,%d]" % (Nmin,Nmax)
                output.write("#Nilsson Diagram for %s, with kappa = %.2f, mu = %.2f\n" % (ntext,par['kappa'],par['mu']) )
                output.write("#delta\t\t")
                for l in range(Nlev):
                    output.write("%s\t" % Nlabel(nQN[l]))
                output.write("\n")
                np.savetxt(output, np.column_stack((deltas,np.transpose(el))),fmt='%.6f',delimiter='\t') 

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
        #### write out to file
        ##if savefilepath is not None:
        ##    with open(savefilepath, "w") as output:
        ##        if Nmax == Nmax:
        ##            ntext = "N = %d" % Nmax
        ##        else:
        ##            ntext = "N = [%d,%d]" % (Nmin,Nmax)
        ##        output.write("#spectroscopic factors for %s, calculated for %s, with kappa = %.2f, mu = %.2f\n" % (Nlabel(nQN[plotorb]),ntext,par['kappa'],par['mu']) )
        ##        output.write("#delta\t\t")
        ##        for l in range(Nlev):
        ##            output.write("%s\t" % Slabel(sQN[l]))
        ##        output.write("\n")
        ##        np.savetxt(output, np.column_stack((deltas,np.transpose(sf))),fmt='%.6f',delimiter='\t') 

    else:
        print("Nilsson level", Nlabel(nQN[plotorb]))

        if Nd==40 and ranged == 0.4:
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
            if Nd==40 and ranged == 0.4:
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
                # write out to file
                if fout is not None:
                    with open(fout, 'w') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerows(zip(deltas,ap[plotorb]))

            if Nd==40 and ranged == 0.4:
                print("\ndecouling parameter")
                for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                    i = np.where(abs(deltas - d)<eps)[0]
                    print("%.4f\t" % ap[plotorb][i],end="")
            
        plt.title("wave function composition for $%s$ level" % Nlabel(nQN[plotorb]))
        if Nd==40 and ranged == 0.4:
            print('\nwave funtion composition')
        plt.plot([deltas[0]-0.2,deltas[-1]+0.2],[0,0],ls="--",linewidth=1,color='k')
        plt.plot([0,0],[np.min(wf[plotorb]),np.max(wf[plotorb])],ls="--",linewidth=1,color='k')
        for l in range(Nlev):
            if max(wf[plotorb][l]) > 0 or min(wf[plotorb][l]) < 0:
                plt.plot(deltas,wf[plotorb][l])
                plt.text(deltas[-1]+0.02,wf[plotorb][l][-1],"$%s$" % (Slabel(sQN[l])), ha = 'left')
                plt.text(deltas[0]-0.02,wf[plotorb][l][0],"$%s$" % (Slabel(sQN[l])), ha = 'right')
                if Nd==40 and ranged == 0.4:
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
            
def test():
    print("testing nothing")
        
if __name__ == "__main__":
   main(sys.argv[1:])

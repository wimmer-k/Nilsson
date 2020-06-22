#!/usr/bin/env python
# for consistent print in python 2 and 3
from __future__ import print_function
import sys
import math
import numpy as np
from wigner import CG
from hamiltonian import Nilsson
import matplotlib.pyplot as plt
import argparse
from diagram import Diagram

version = "0.4/2020.06.22"

# constants
eps = 0.001 # tolerance for delta matching
#display stuff
textshift = 0.01


# todo:
# FIXED - spherical quantum numbers are wrong
# DONE - calculate decoupling parameters, needs j
# DONE - calculate magnitc moments, g-factors
# DONE - calculate spectroscopic factors
# - g-factor for states with not Omega = K = J needs further options
# - parameters change as function of N, not nexcessarly Nmax, how to deal with this
# DONE - parameters, states and space are returned by runnilsson, but that is not required if the space is part of input
# DONE - commandline interface
# DONE - test with python 3
# DONE - separate plotting function
# DONE - put calculation into separate class
# DONE option to calculate only a range of N = [Nmin,Nmax]
# - if not the whole diagram is needed calculate only the states of this Omega and parity (for gui the wave function is stored.)
# DONE option to change kappa and mu from standard values.
# DONE output g_K instead / in addition
# DONE in the option to calculate a, also output the magnetic decoupling b, done, but needs checking


def main(argv):
    
    np.set_printoptions(linewidth=300)
    
    # defaults
    verbose = False
    noplot = False 
    Nmin = 3
    Nmax = 3
    plotorb = -1
    plotopt = {'diagram': True, 'wavef': False, 'decoup': False, 'gfact': False, 'sfact': False}
    ranged = 0.4
    Nd = 40
    kappa = 0.05
    mu = -1
    quench = 0.9
    gR = 0.35
    savefilepath = None
    DeltaN2 = True
    
    argparser = argparse.ArgumentParser(description='Calculation of Nilsson diagrams and wave functions.', add_help=False)
    requiredargs = argparser.add_argument_group('required arguments')
    requiredargs.add_argument("-N", "-n", "--nosc", required=True,nargs="+",dest="Nosc",type=int,help="oscillator shell N, or a range of oscillator shells Nmin Mmax, e.g. -N 3 or -N 1 3")

    argparser.add_argument("-o", "--orbital", dest="orb", type=int,help="number of orbital to be plotted")
    argparser.add_argument("-p", "--property", dest="prop", choices=['wavef','wf','decoup','a', 'gfactor','gfact','g', 'sfactor','sfact','s'], help="which property should be plotted. The options are wave function: \"wavef\" or \"wf\", decoupling parameter: \"decoup\" or \"a\", g-factor: \"gfactor\", \"gfact\", or \"g\", spectroscopic factors: \"sfactor\", \"sfact\", or \"s\"")
    
    argparser.add_argument("-k","--kappa", dest="kappa", type=float,help="kappa value for the energy calculations, default value is 0.05")
    argparser.add_argument("-m","--mu", dest="mu", type=float,help="mu value for the energy calculations, default value depend on the value of N")
    argparser.add_argument("-Nd","--ndelta", dest="Nd", type=int,help="number of steps in delta, typical value is 40")
    argparser.add_argument("-r","--ranged", dest="ranged", type=float,help="range of delta values [-ranged,ranged], typical value is 0.4")
    argparser.add_argument("-gR","--gfactR", dest="gfactR", type=float,help="g factor options, parameter g_R, typically Z/A")
    argparser.add_argument("-q","--gquench", dest="gquench", type=float,help="g factor options, quenching of spin g-factor, default 0.7")
    argparser.add_argument("-w","--write", dest="write", type=str,help="write the calculation to output file")
    argparser.add_argument("-noplot","--noplot",help="do not plot", action="store_true")
    argparser.add_argument("-ndN2","--no-deltaN2",dest="nodeltaN2",help="disable deltaN=2 coupling", action="store_true")
    
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
        verbose = True
    if args.noplot:
        noplot = True
    if args.nodeltaN2:
        DeltaN2 = False
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
        else:
            plotopt['wavef'] = True

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
    if args.gfactR is not None:
        if args.gfactR < 0:
            raise argparser.error("-g/--gfact must be positive, typical value gR = Z/A, default is 0.35")
        else:
            gR = args.gfactR
    if args.gquench is not None:
        if args.gquench < 0:
            raise argparser.error("-g/--gquench must be positive, default is 0.70")
        else:
            quench = args.gquench
        
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

    print(DeltaN2)
    diag = Diagram(kappa=par['kappa'], mu=par['mu'], Nd=diaopt['Nd'], ranged=diaopt['maxd'], Nmin = Nmin, Nmax = Nmax, DeltaN2=DeltaN2, verbose=verbose)
    diag.rundiagram()

        
    ## plotting    
    ax = []
    if plotopt['diagram']:
        fig = plt.figure(figsize=[6,9])
        ax.append(plt.subplot(1,1,1))
        if Nmin == Nmax:
            plt.title("Nilsson Diagram for $N = %d$, $\kappa$ = %.2f, $\mu$ = %.2f" % (Nmax,par['kappa'],par['mu']))
        else:
            plt.title("Nilsson Diagram $N = [%d,%d]$, $\kappa$ = %.2f, $\mu$ = %.2f" % (Nmin,Nmax,par['kappa'],par['mu']))
        plt.plot([0,0],[np.min(diag.el),np.max(diag.el)],ls="--",linewidth=1,color='k')
        for l in range(diag.Nlev):
            plt.plot(diag.deltas,diag.el[l], ls="--" if diag.nQN[l][1] == 1 else "-")
            plt.text(diag.deltas[-1]+textshift,diag.el[l][-1],"$%s$, %d" % (diag.Nlabel(diag.nQN[l]),l), ha = 'left')
            plt.text(diag.deltas[0]-textshift,diag.el[l][0],"$%s$, %d" % (diag.Nlabel(diag.nQN[l]),l), ha = 'right')
        plt.ylabel('$E/\hbar\omega$')
        ## write out to file
        if savefilepath is not None:
            with open(savefilepath, "w") as output:
                if Nmin == Nmax:
                    ntext = "N = %d" % Nmax
                else:
                    ntext = "N = [%d,%d]" % (Nmin,Nmax)
                output.write("#Nilsson Diagram for %s, with kappa = %.2f, mu = %.3f\n" % (ntext,par['kappa'],par['mu']) )
                output.write("#delta\t\t")
                for l in range(diag.Nlev):
                    output.write("%s\t" % diag.Nlabel(diag.nQN[l]))
                output.write("\n")
                np.savetxt(output, np.column_stack((diag.deltas,np.transpose(diag.el))),fmt='%.6f',delimiter='\t')
    else:
        print("delta ")
        for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
            print("%.3f\t" % d,end="")
        
    if plotopt['wavef']:
        fig = plt.figure(figsize=[6,6])
        ax.append(plt.subplot(1,1,1))
        rwf, rqn = diag.wavefunction(plotorb)
        plt.title("wave function composition for $%s$ level" % diag.Nlabel(diag.nQN[plotorb]))
        if Nd==40 and ranged == 0.4:
            print('\nwave funtion composition')
        plt.plot([0,0],[np.min(rwf),np.max(rwf)],ls="--",linewidth=1,color='k')
        plt.plot([diag.deltas[0]-0.1,diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color='k')
        for l in range(len(rqn)):
            plt.plot(diag.deltas,rwf[l])
            plt.text(diag.deltas[-1]+textshift,rwf[l][-1],"$%s$" % (diag.Slabel(rqn[l])), ha = 'left')
            plt.text(diag.deltas[0]-textshift,rwf[l][0],"$%s$" % (diag.Slabel(rqn[l])), ha = 'right')
            if Nd==40 and ranged == 0.4:
                for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                    i = np.where(abs(diag.deltas - d)<eps)[0]
                    print("%.4f\t" % rwf[l][i],end="")
                print(diag.Slabel(rqn[l]))
        plt.ylabel('$c_{\Omega j}$')
        if savefilepath is not None:
            with open(savefilepath, "w") as output:
                if Nmin == Nmax:
                    ntext = "N = %d" % Nmax
                else:
                    ntext = "N = [%d,%d]" % (Nmin,Nmax)
                output.write("#wave function composition for %s level, calculated for %s, with kappa = %.2f, mu = %.3f\n" % (diag.plainNlabel(diag.nQN[plotorb]), ntext, par['kappa'], par['mu']) )
                output.write("#delta\t\t")
                for l in range(len(rqn)):
                    output.write("%s\t\t" % diag.plainSlabel(rqn[l]))
                output.write("\n")
                np.savetxt(output, np.column_stack((diag.deltas,np.transpose(rwf))),fmt='%.6f',delimiter='\t') 
    
    # check if this is K = 1/2
    if plotopt['decoup'] and diag.nQN[plotorb][0] == 1./2:
        a = diag.decoupling(plotorb)
        sz, gKN, gKP, gN, gP, bN, bP = diag.gfactors(plotorb,quench,gR)       

        fig = plt.figure(figsize=[6,6])
        ax.append(plt.subplot(2,1,1))
        if type(a) is not np.ndarray:
            return
        plt.title("decoupling parameters for $%s$ level" % diag.Nlabel(diag.nQN[plotorb]))
        plt.plot([0,0],[np.min(a),np.max(a)],ls="--",linewidth=1,color='k')
        plt.plot(diag.deltas,a)
        plt.ylabel('$a$')
        plt.setp(ax[-1].get_xticklabels(), visible=False)
        ax[-1].tick_params(axis='y', which='major')
        ax[-1].tick_params(axis='y', which='minor')

        ax.append(plt.subplot(2,1,2, sharex = ax[0]))
        mi = np.min([np.min(bN),np.min(bP)])
        ma = np.max([np.max(bN),np.max(bP)])

        ax[-1].plot([0,0],[mi,ma],ls="--",linewidth=1,color='k')
        ax[-1].plot(diag.deltas,bN,label="neutron")
        ax[-1].plot(diag.deltas,bP,label="proton")
        ax[-1].tick_params(axis='both', which='major')
        ax[-1].tick_params(axis='both', which='minor')
        ax[-1].set_ylabel('$b_{0}$',fontsize=12)
        
        plt.legend(loc="best",fancybox=False,fontsize=12)
                                    
        if Nd==40 and ranged == 0.4:
            print("\ndecouling parameter")
            for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                i = np.where(abs(diag.deltas - d)<eps)[0]
                print("%.4f\t" % a[i],end="")
            print()
 
        if savefilepath is not None:
            with open(savefilepath, "w") as output:
                if Nmin == Nmax:
                    ntext = "N = %d" % Nmax
                else:
                    ntext = "N = [%d,%d]" % (Nmin,Nmax)
                output.write("#decoupling parameters for %s level, calculated for %s, with kappa = %.2f, mu = %.3f\n" % (diag.Nlabel(diag.nQN[plotorb]), ntext, par['kappa'], par['mu']) )
                output.write("#delta\t\ta\t\tbN\t\tbP\n")
                np.savetxt(output, np.column_stack((diag.deltas,np.transpose(a),np.transpose(bN),np.transpose(bP))),fmt='%.6f',delimiter='\t') 

    if plotopt['gfact']:
        fig = plt.figure(figsize=[8,9])
        if diag.nQN[plotorb][0] == 1./2:
            ax.append(plt.subplot2grid((4,1),(0,0)))
            ax.append(plt.subplot2grid((4,1),(1,0),sharex=ax[0]))
            ax.append(plt.subplot2grid((4,1),(2,0),sharex=ax[0]))
            ax.append(plt.subplot2grid((4,1),(3,0),sharex=ax[0]))
        else:
            ax.append(plt.subplot2grid((3,1),(0,0)))
            ax.append(plt.subplot2grid((3,1),(1,0),sharex=ax[0]))
            ax.append(plt.subplot2grid((3,1),(2,0),sharex=ax[0]))
        sz, gKN, gKP, gN, gP, bN, bP = diag.gfactors(plotorb,quench,gR)        
        #fN = b0N*(gKN-gR)
        #fP = b0P*(gKP-gR)
        fN = bN*(gKN-gR)
        fP = bP*(gKP-gR)
        
        if type(sz) is not np.ndarray:
            return
        ax[0].set_title("$g$-factor for $%s$ level" % diag.Nlabel(diag.nQN[plotorb]))
        ax[0].plot([0,0],[np.min(sz),np.max(sz)],ls="--",linewidth=1,color='k')
        ax[0].plot([diag.deltas[0]-0.1,diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color='k')
        plt.setp(ax[0].get_xticklabels(), visible=False)
        ax[0].plot(diag.deltas,sz)
        ax[0].set_ylabel('$<s_z>$')
        
        mi = np.min([np.min(gKN),np.min(gKP)])
        ma = np.max([np.max(gKN),np.max(gKP)])
        ax[1].plot([0,0],[mi,ma],ls="--",linewidth=1,color='k')
        ax[1].plot([diag.deltas[0]-0.1,diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color='k')
        ax[1].plot(diag.deltas,gKN,label="neutron")
        ax[1].plot(diag.deltas,gKP,label="proton")
        plt.setp(ax[1].get_xticklabels(), visible=False)
        ax[1].tick_params(axis='y', which='major')
        ax[1].tick_params(axis='y', which='minor')
        ax[1].set_ylabel('$g_K$',fontsize=12)
        
        mi = np.min([np.min(gN),np.min(gP)])
        ma = np.max([np.max(gN),np.max(gP)])
        ax[2].plot([0,0],[mi,ma],ls="--",linewidth=1,color='k')
        ax[2].plot([diag.deltas[0]-0.1,diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color='k')
        ax[2].plot(diag.deltas,gN,label="neutron")
        ax[2].plot(diag.deltas,gP,label="proton")
        ax[2].set_xlim(diag.deltas[0]-0.1, diag.deltas[-1]+0.1)
        plt.setp(ax[2].get_xticklabels(), visible=False)
        ax[2].tick_params(axis='y', which='major')
        ax[2].tick_params(axis='y', which='minor')
        ax[2].set_ylabel('$g$',fontsize=12)

        if diag.nQN[plotorb][0] == 1./2:
            mi = np.min([np.min(fN),np.min(fP)])
            ma = np.max([np.max(fN),np.max(fP)])
            ax[3].plot([0,0],[mi,ma],ls="--",linewidth=1,color='k')
            ax[3].plot([diag.deltas[0]-0.1,diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color='k')
            ax[3].plot(diag.deltas,fN,label="neutron")
            ax[3].plot(diag.deltas,fP,label="proton")
            ax[3].set_ylabel('$b_0(g_K-g_R)$',fontsize=12)
            
        if Nd==40 and ranged == 0.4:
            print("\ng-factors")
            for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                i = np.where(abs(diag.deltas - d)<eps)[0]
                print("%.4f\t" % gN[i],end="")
            print("neutron")
            for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                i = np.where(abs(diag.deltas - d)<eps)[0]
                print("%.4f\t" % gP[i],end="")
            print("proton")
            
        if savefilepath is not None:
            with open(savefilepath, "w") as output:
                if Nmin == Nmax:
                    ntext = "N = %d" % Nmax
                else:
                    ntext = "N = [%d,%d]" % (Nmin,Nmax)
                output.write("#g-factor for %s level, calculated for %s, with kappa = %.2f, mu = %.3f\n" % (diag.plainNlabel(diag.nQN[plotorb]), ntext, par['kappa'], par['mu']) )
                output.write("#delta\t\tsz\t\tgKN\t\tgKP\t\tgN\t\tgP\t\tbN\t\tbP\t\tbN(gK-gR)\t\tbP(gK-gR)\n")
                np.savetxt(output, np.column_stack(( diag.deltas, np.transpose(sz), np.transpose(gKN), np.transpose(gKP),
                                                     np.transpose(gN), np.transpose(gP),
                                                     np.transpose(bN), np.transpose(bP), np.transpose(fN), np.transpose(fP) ))
                           ,fmt='%.6f',delimiter='\t') 
                
    if plotopt['sfact']:
        fig = plt.figure(figsize=[6,6])
        ax.append(plt.subplot(2,1,1))
        rsf, rqn = diag.sfactors(plotorb)
        plt.title("spectroscopic factors for $%s$ level" % diag.Nlabel(diag.nQN[plotorb]))
        if Nd==40 and ranged == 0.4:
            print('\nspectroscopic factors')
        plt.plot([0,0],[0.0,np.max(rsf)],ls="--",linewidth=1,color='k')
        for l in range(len(rqn)):
            plt.plot(diag.deltas,rsf[l])
            plt.text(diag.deltas[-1]+textshift,rsf[l][-1],"$%s$" % (diag.Slabel(rqn[l])), ha = 'left')
            plt.text(diag.deltas[0]-textshift,rsf[l][0],"$%s$" % (diag.Slabel(rqn[l])), ha = 'right')
            if Nd==40 and ranged == 0.4:
                for d in np.linspace(-0.4,0.4,num=9, endpoint=True):
                    i = np.where(abs(diag.deltas - d)<eps)[0]
                    print("%.4f\t" % rsf[l][i],end="")
                print(diag.Slabel(rqn[l]))
        plt.ylabel('$|c_{\Omega j}|^2$')
        plt.setp(ax[-1].get_xticklabels(), visible=False)
        ax[-1].tick_params(axis='y', which='major')
        ax[-1].tick_params(axis='y', which='minor')


        ax.append(plt.subplot(2,1,2, sharex = ax[0]))
        """ from Ii = 0 to any orbital, nucleon adding """
        g2 = 2
        maxsf = 0
        for l in range(len(rqn)):
            jfact = 1./(2*rqn[l][2]+1)
            dK = diag.nQN[plotorb][0]
            j = rqn[l][2]
            If = rqn[l][2]
            Kf = diag.nQN[plotorb][0]
            cg = CG(0,0,j,dK,If,Kf)
            sf = jfact*g2*cg**2*rsf[l]
            maxsf = max(np.max(sf),maxsf)
            ax[1].plot(diag.deltas,sf)
            ax[1].text(diag.deltas[-1]+textshift,sf[-1],"$%s$" % (diag.Slabel(rqn[l])), ha = 'left')
            ax[1].text(diag.deltas[0]-textshift,sf[0],"$%s$" % (diag.Slabel(rqn[l])), ha = 'right')
            ax[-1].tick_params(axis='both', which='major')
            ax[-1].tick_params(axis='both', which='minor')
            ax[-1].set_ylabel(r'$S(0^+\rightarrow j)$',fontsize=12)

        
        if savefilepath is not None:
            with open(savefilepath, "w") as output:
                if Nmin == Nmax:
                    ntext = "N = %d" % Nmax
                else:
                    ntext = "N = [%d,%d]" % (Nmin,Nmax)
                output.write("#spectroscopic factors for %s level, calculated for %s, with kappa = %.2f, mu = %.3f\n" % (diag.plainNlabel(diag.nQN[plotorb]), ntext, par['kappa'], par['mu']) )
                output.write("#delta\t\t")
                for l in range(len(rqn)):
                    output.write("%s\t\t" % diag.plainSlabel(rqn[l]))
                output.write("\n")
                np.savetxt(output, np.column_stack((diag.deltas,np.transpose(rsf))),fmt='%.6f',delimiter='\t') 

    plt.xlabel('$\delta$')
    plt.tick_params(axis='both', which='major')
    plt.tick_params(axis='both', which='minor')
    plt.xlim(diag.deltas[0]-0.2, diag.deltas[-1]+0.2)
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    fig.align_ylabels(ax)
    if not noplot:
        plt.show()

def test():
    print("testing nothing")
        
if __name__ == "__main__":
   main(sys.argv[1:])

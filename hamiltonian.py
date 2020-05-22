#!/usr/bin/env python
# for consistent print in python 2 and 3
from __future__ import print_function
from memoize import Memoize
from wigner import CG
import numpy as np

verbose = False # True #
defaultmu = [0,0,0,0.35,0.45,0.45,0.45,0.40]
class Nilsson(object):
    def __init__(self,*args,**kwargs):
        self.Nmin = kwargs.get('Nmin',0)
        self.Nmax = kwargs.get('Nmax',3)
        self.Omega = kwargs.get('Omega',1.5)
        self.Parity = kwargs.get('Parity',1)
        self.DeltaN2 = kwargs.get('DeltaN2',True)

        self.Verbose = kwargs.get('Verbose',False)
        self.states = self.createstates()
        self.par ={'kappa': 0.05, 'mu': defaultmu[self.Nmax], 'delta': 0.0}
                
    """ creates all states until Nmax with a certain Omega and Parity """
    def createstates(self):
        states =[]
        if self.Nmin % 2 == 0:
            Nsta = self.Nmin+self.Parity
        else:
            Nsta = self.Nmin+(1-self.Parity)
            
        # 0,2,4... for positive parity, 1,3,5,... for negative
        for n in range(Nsta,self.Nmax+1,2):
            for l in range(self.Parity,n+1,2): #checks for parity
                for ml in range(-l,l+1): 
                    for ms in [-1./2,+1./2]: #ms
                        if self.Omega==ml+ms:
                            states.append((n,l,ml,ms))
        return states

    def setparameters(self,*args,**kwargs):
        defmu = [0,0,0,0.35,0.45,0.45,0.45,0.40]
        self.par['kappa'] = kwargs.get('kappa',0.05)
        self.par['mu'] = kwargs.get('mu',defaultmu[self.Nmax])
        self.par['delta'] = kwargs.get('delta',0.3)

    """ calculate the eigenvalues and vectors for a set of parameters, input model space """
    def calculate(self,delta):
        self.par['delta'] = delta
        h = self.hamiltonian()
        val, vec = self.diagonalize(h)
        return val, vec

    """ create the Hamiltonian matrix """
    def hamiltonian(self):
    
        nmax = self.Nmax
        omega = self.Omega
        pairty = self.Parity
        kappa = self.par['kappa']
        mu =    self.par['mu']
        delta = self.par['delta']

        nstates = len(self.states)
        
        # for volume conservation wx*wy*wz = w^3
        omega_d = pow(1 - 4./3*delta**2 -16./27*delta**3 , -1./6)
    
        if self.Verbose:
            print("delta = %.2f, kappa = %.2f, mu = %.2f, omega = %.2f" %(delta,kappa,mu,omega_d))
        
        # Hamiltonian matrix
        H = np.zeros(shape=(nstates,nstates))
        for i in range(nstates):
            N,l,ml,ms = self.states[i]
            # diagonal 
            #        H_0                  # l^2     # <l^2>         # spin orbit            (<l^2> Nilsson/Ragnarson Excersise 6.7, )
            H[i,i] = N + 3./2 - kappa*mu*(l*(l+1) - 1./2*N*(N+3)) - kappa*2*ml*ms
            # using 3/2 and 1/2 makes the code just a touch slower, but the symbolical notation is preferred
            # off diagonal
            for j in range(i+1,nstates):
                N2,l2,ml2,ms2 = self.states[j]
                if N!=N2:
                    continue
                if l!=l2:
                    continue
                if ml+ms != ml2+ms2:
                    continue
                #print(N,l,ml,ms,"\t",N2,l2,ml2,ms2)
                for sign in [-1.,+1.]:
                    if ml2 == ml+sign and ms2 == -sign*1/2 and ms == sign*1/2:
                        # spin orbit-part
                        H[i,j] = -2*kappa* 1/2*np.sqrt( (l-sign*ml)*(l+sign*ml+1) )
                H[j,i] = H[i,j]

        Hd = self.hamiltonian_delta
        if self.Verbose:
            print("H0 = \n", H)
            print("delta(omega) = ",-delta*omega_d*4./3*np.sqrt(np.pi/5))
            print("Hd = \n", Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5)))
            print("H = \n", H + Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5)))
        
        return H + Hd*(-delta*omega_d*4./3*np.sqrt(np.pi/5))


    #deformation part of the Hamiltonian, this should be stored to save time when many deformations are calculated
    @Memoize
    def hamiltonian_delta(self):
        nstates = len(self.states)
        H = np.zeros(shape=(nstates,nstates))
        for i in range(nstates):
            N,l,ml,ms = self.states[i]
            for j in range(i,nstates):
                N2,l2,ml2,ms2 = self.states[j]
                H[i,j] = self.rhoY20(N,l,ml,ms,N2,l2,ml2,ms2)
                H[j,i] = H[i,j]
        return H
    
    def rhoY20(self, N,l,ml,ms,N2,l2,ml2,ms2):
        # selection rules
        if ms!=ms2 or ml!=ml2:
            return 0
        if not (N==N2 or abs(N-N2)==2):
            return 0
        # option to include the DeltaN=2 admixtures
        if not self.DeltaN2 and N!=N2: 
            return 0
        if not (l==l2 or abs(l-l2)==2):
            return 0
        return self.r2(N,l,N2,l2) * self.Y20(l,ml,l2,ml2)

    def Y20(self, l,ml,l2,ml2):
        # <l'm'|Y20|l,m>
        return np.sqrt( (5./4/np.pi) * (2*l+1.)/(2*l2+1) ) * CG(l,ml,2,0, l2, ml2) * CG(l,0,2,0,l2,0) 

    def r2(self, N,l,N2,l2):
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

    # diagonalize the Hamiltonian, returns eigenvalues and eigenvectors
    def diagonalize(self, ham):
        eValues, eVectors = np.linalg.eig(ham)
        idx = eValues.argsort() 
        eValues = eValues[idx]
        eVectors = eVectors[:,idx]
        if self.Verbose:
            print("idx", idx)
            print("eValues\n", eValues)
            print("eVectors\n", eVectors)
            print("eVectors^T\n", np.transpose(eVectors))
        # eigenvectors are columns, therefore transpose
        return eValues, np.transpose(eVectors)

    # determine the wave functions in terms of the spherical states
    def wavefunction(self, trafo, evectors):
        return np.array([np.dot(trafo,v) for v in evectors])

    # calculate basis transformation
    def basistrafo(self, evectors):
        #print evectors
        mm = np.transpose(evectors)
        if self.Verbose:
            print("matrix of eigenvectors")
            print(mm)
            print("inverted matrix of eigenvectors")
            print(np.linalg.inv(mm))
        return np.linalg.inv(mm)


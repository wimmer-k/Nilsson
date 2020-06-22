from hamiltonian import Nilsson
import numpy as np

ell = ["s","p","d","f","g","h","i","j"]
parities = ["+","-"]
eps = 0.001 # tolerance for energy matching
# g factors free proton, neutron
gs = [5.585694, -3.826085]
gl = [1.0     ,  0.0     ]

class Diagram():
    def __init__(self,*args,**kwargs):
        self.par = {'kappa': kwargs.get('kappa',0.05), 'mu': kwargs.get('mu',0.35),
                    'Nd': kwargs.get('Nd',40), 'mind': -kwargs.get('ranged',0.4), 'maxd': kwargs.get('ranged',0.4),
                    'Nmin': kwargs.get('Nmin',2), 'Nmax': kwargs.get('Nmax',2),
                    'DeltaN2': kwargs.get('DeltaN2',True)}
        
        self.verbose = kwargs.get('verbose',False)


        """ Basic model space and parameters setup """
        Nnuc = (self.par['Nmax']+1)*(self.par['Nmax']+2)*(self.par['Nmax']+3)/3 - (self.par['Nmin']-1+1)*(self.par['Nmin']-1+2)*(self.par['Nmin']-1+3)/3
        self.Nlev = int(Nnuc/2)
        
        if self.verbose:
            print("number of Nilsson levels:", self.Nlev)
            
        self.deltas = np.linspace(self.par['mind'],self.par['maxd'],num=self.par['Nd']*2+1,endpoint=True)
        
        #storage containers for el = energy levels, wf = wave functions (c), ev = eigenvectors (a)
        self.el = np.zeros((self.Nlev,self.par['Nd']*2+1))
        self.wf = np.zeros((self.Nlev,self.Nlev,self.par['Nd']*2+1))
        self.ev = np.zeros((self.Nlev,self.Nlev,self.par['Nd']*2+1))
        # states for the calculation N,l,ml,ms
        self.QN = [() for r in range(self.Nlev)]
        # Nilsson quantum numbers
        self.nQN = [() for r in range(self.Nlev)]
        # quantum numbers of the spherical states (origin)
        self.sQN = [() for r in range(self.Nlev)]

    def rundiagram(self):
        """ Run the Nilsson calculation for a range of delta values and for N = [N_min,N_max] """
        Nd = self.par['Nd']
        cur = 0
        for Omega in np.arange(1./2,self.par['Nmax']+1,1):
            if self.verbose:
                print("Omega ", Omega)
            for Parity in [0,1]:
                if self.verbose:
                    print("Parity ", Parity)

                nilsson = Nilsson(Nmin = self.par['Nmin'], Nmax = self.par['Nmax'], Omega = Omega, Parity = Parity, DeltaN2=self.par['DeltaN2'], Verbose=self.verbose)
                nilsson.setparameters(**self.par)
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
                    E = N+3./2-self.par['kappa']*self.par['mu']*(l*(l+1)- 1./2*N*(N+3))-2*self.par['kappa']*SO*1.0/2
                    o = int(np.where(abs(E-spval)<eps)[0])
                    slabels[o] = (n,l,j,Parity)


                """ calculated from delta = 0 out to check for sign jumps in the wave function """
                #oblate part
                for d in range(Nd+1):
                    delta = self.deltas[Nd-d]
                    val, vec = nilsson.calculate(delta)
                    wfcoeff = nilsson.wavefunction(bt,vec)
                    
                    for v in range(nstates):
                        self.el[cur+v][Nd-d] = val[v]
                        # check overlap with last calculation, to avoid jumps invert vectors
                        if d > 0 and np.dot(lastwfcoeff[v],wfcoeff[v]) < np.dot(lastwfcoeff[v],-wfcoeff[v]):
                            wfcoeff[v] = -wfcoeff[v]
                        for w in range(len(wfcoeff[v])):
                            self.wf[cur+v,cur+w,Nd-d] = wfcoeff[v][w]
                            self.ev[cur+v,cur+w,Nd-d] = vec[v][w]
                            
                    lastwfcoeff = wfcoeff

                #reset last coefficients
                lastwfcoeff = nilsson.wavefunction(bt,spvect)

                #prolate part
                for d in range(Nd+1):
                    delta = self.deltas[Nd+d]
                    val, vec = nilsson.calculate(delta)
                    wfcoeff = nilsson.wavefunction(bt,vec)
                    
                    for v in range(nstates):
                        self.el[cur+v][Nd+d] = val[v]
                        # check overlap with last calculation, to avoid jumps invert vectors
                        if d > 0 and np.dot(lastwfcoeff[v],wfcoeff[v]) < np.dot(lastwfcoeff[v],-wfcoeff[v]):
                            wfcoeff[v] = -wfcoeff[v]
                        for w in range(len(wfcoeff[v])):
                            self.wf[cur+v,cur+w,Nd+d] = wfcoeff[v][w]
                            self.ev[cur+v,cur+w,Nd+d] = vec[v][w]

                    lastwfcoeff = wfcoeff

                """ asymptotic quantum numbers and spherical labels """
                ctr = [0 for c in range(self.par['Nmax']+1)]
                for v in range(nstates):
                    N = states[v][0]
                    nz = N-ctr[N]-(Omega-1./2)
                    l = states[v][1]
                    Lambda = -1
                    if (Omega - 1./2 +nz)%2 == N%2:
                        Lambda = Omega - 1./2
                    if (Omega + 1./2 +nz)%2 == N%2:
                        Lambda = Omega + 1./2
                    if self.verbose:
                        print("N = ", N ,", l =",  l ,", ml = ",  states[v][2] ,", ms = ", states[v][3] ,", v = ", v , ", nz = ", nz, ", Lamba = ", Lambda, ", Omega = ", Omega)
                    self.nQN[cur+v] = (Omega,Parity,N,nz,Lambda)
                    self.sQN[cur+v] = slabels[v]
                    self.QN[cur+v] = (N,l,states[v][2],states[v][3])
                    ctr[N] = ctr[N] +1
                
                if nstates > 0:
                    cur = cur + nstates

                    
    def wavefunction(self,o):
        """ Remove empty components, adjust the labels, for plotting and saving """
        reswf = self.wf[o][np.any(self.wf[o] != 0, axis=1), :]
        ind = np.any(self.wf[o] != 0, axis=1)
        ind = [i for i,x in enumerate(ind) if x]
        ressQN = [self.sQN[i] for i in ind]
        return reswf, ressQN

    def decoupling(self,o):
        """ Calculate the decoupling parameters """
        if self.nQN[o][0] != 1./2:
            print("decoupling parameter only for Omega = 1/2")
            return None
        a = np.zeros(self.par['Nd']*2+1)
        for w in range(len(self.wf[o])):
            a += pow(-1,self.sQN[w][2]-1./2)*(self.sQN[w][2]+1./2)*self.wf[o][w]**2
            
        ##""" alternative wave to calculate the decoupling parameter a' using the l,m_l basis """        
        ##ap = np.zeros(self.par['Nd']*2+1)
        ##al0 = 0
        ##al1 = 0
        ##for w in range(len(self.ev[o])):
        ##    if self.QN[w][2] == 0 and self.QN[w][3] == 0.5:
        ##        al0 = self.ev[o][w]
        ##        ap += self.ev[o][w]**2
        ##    if self.QN[w][2] == 1 and self.QN[w][3] == -0.5:
        ##        al1 = self.ev[o][w]
        ##        ap += 2*np.sqrt(self.QN[w][1]*(self.QN[w][1]+1))*al0*al1
        ##ap *=  pow(-1,self.nQN[o][1])
        
        return a
                
    def gfactors(self,o,quench,gR):
        """ Calculate the g factors, and <s_z> """
        Omega = self.nQN[o][0]
        #so far only for I = K = Omega
        I = Omega
        K = Omega
        
        sz = np.zeros(self.par['Nd']*2+1)
        jz = np.zeros(self.par['Nd']*2+1)
        sum_a_ml0 = np.zeros(self.par['Nd']*2+1)
        for w in range(len(self.ev[o])):
            sz += self.QN[w][3]*self.ev[o][w]**2*pow(-1,Omega-1./2)
            jz += Omega*self.ev[o][w]**2*pow(-1,Omega-1./2)
            if Omega == 1./2 and self.QN[o][2] == 0:
                sum_a_ml0 += pow(-1,I-1./2+self.QN[w][1])*self.ev[o][w]**2

        gKN = (gs[1]*quench-gl[1] )*sz/jz + gl[1]
        gKP = (gs[0]*quench-gl[0] )*sz/jz + gl[0]

        gN = K*Omega/(I*(I+1))*(gKN-gR) + gR
        gP = K*Omega/(I*(I+1))*(gKP-gR) + gR
        if Omega == 1./2:
            a = self.decoupling(o)
            gN += (gl[1]-gR)*pow(-1,I+1./2) *1./2 *(I+1./2) * a + \
                + (gs[1]*quench-gl[1]) *1./2 *(I+1./2) * sum_a_ml0
            gP += (gl[0]-gR)*pow(-1,I+1./2) *1./2 *(I+1./2) * a + \
                + (gs[0]*quench-gl[0]) *1./2 *(I+1./2) * sum_a_ml0
            bN = (-(gl[1]-gR)*a -1./2*pow(-1, self.QN[w][1])*(gs[1]*quench+gKN-2*gl[1])) / (gKN-gR)
            bP = (-(gl[0]-gR)*a -1./2*pow(-1, self.QN[w][1])*(gs[0]*quench+gKP-2*gl[0])) / (gKP-gR)

            
            ##""" magnetic decoupling paramter b alternative"""
            ##bN = np.zeros(self.par['Nd']*2+1)
            ##bP = np.zeros(self.par['Nd']*2+1)
            ##al0 = 0
            ##al1 = 0
            ##for w in range(len(self.ev[o])):
            ##    if self.QN[w][2] == 0 and self.QN[w][3] == 0.5:
            ##        al0 = self.ev[o][w]
            ##        bN += self.ev[o][w]**2*(gs[1]*quench-gR)
            ##        bP += self.ev[o][w]**2*(gs[0]*quench-gR)
            ##    if self.QN[w][2] == 1 and self.QN[w][3] == -0.5:
            ##        al1 = self.ev[o][w]
            ##        bN += 2*(gl[1]-gR)*np.sqrt(self.QN[w][1]*(self.QN[w][1]+1))*al0*al1
            ##        bP += 2*(gl[0]-gR)*np.sqrt(self.QN[w][1]*(self.QN[w][1]+1))*al0*al1
            ##bN *=  -pow(-1,self.nQN[o][1])/(gKN-gR)
            ##bP *=  -pow(-1,self.nQN[o][1])/(gKP-gR)

            
        else:
            bN = np.zeros(self.par['Nd']*2+1)
            bP = np.zeros(self.par['Nd']*2+1)
            
        return sz, gKN, gKP, gN, gP, bN, bP
            
    def sfactors(self,o):
        """ Calculate the spectroscopic factors """
        """ Remove empty components, adjust the labels, for plotting and saving """
        reswf = self.wf[o][np.any(self.wf[o] != 0, axis=1), :]
        ind = np.any(self.wf[o] != 0, axis=1)
        ind = [i for i,x in enumerate(ind) if x]
        ressQN = [self.sQN[i] for i in ind]
        return reswf**2, ressQN

        
    def Nlabel(self,QN):
        """ Label with the Nilsson quantum numbers, formatted for Latex type printing """
        Omega, Parity, N, nz, Lambda = QN
        return "%d/2^{%s}[%d%d%d]" % (Omega*2,parities[Parity],N,nz,Lambda)

    def Slabel(self,QN):
        """ Label with the spherical quantum numbers, formatted for Latex type printing """
        n, l, j, Parity = QN
        return "%d%s_{%d/2}" % (n,ell[l],2*j)

    def plainNlabel(self,QN):
        """ Label with the Nilsson quantum numbers, plain text """
        Omega, Parity, N, nz, Lambda = QN
        return "%d/2%s[%d%d%d]" % (Omega*2,parities[Parity],N,nz,Lambda)

    def plainSlabel(self,QN):
        """ Label with the spherical quantum numbers, plain text """
        n, l, j, Parity = QN
        return "%d%s%d/2" % (n,ell[l],2*j)


import numpy as np

from .utils import *
from .constants import *

class model_potential:
    """
    Class to represent model potential for use in python numerov wavefunction calculations.
    This code is adapted directly from the Alkali Rydberg Calculator (ARC).
    """
    
    def __init__(self,alphaC,a1,a2,a3,a4,rc,Z,include_so=True, use_model = True):
        #FIXME -- this construction and its takes parameters for up to l=3, but this is implicit and should be made explicit

            self.alphaC = alphaC
            self.a1 = a1
            self.a2 = a2
            self.a3 = a3
            self.a4 = a4
            self.rc = rc
            
            self.Z = Z
            
            self.include_so = include_so
            self.use_model = use_model
    
    def core_potential(self, ch, r):
        """ FROM ARC """
        """ core potential felt by valence electron
            For more details about derivation of model potential see
            Ref. [#marinescu]_.
            Args:
                l (int): orbital angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: core potential felt by valence electron (in a.u. ???)
            References:
                .. [#marinescu] M. Marinescu, H. R. Sadeghpour, and A. Dalgarno
                    PRA **49**, 982 (1994),
                    https://doi.org/10.1103/PhysRevA.49.982
        """
        l = ch.l
        return -self.effective_charge(ch, r) / r - self.alphaC / (2 * r**4) * \
            (1 - np.exp(-(r / self.rc[l])**6))
    
    def effective_charge(self, ch, r): #l, r):
        """ effective charge of the core felt by valence electron
            For more details about derivation of model potential see
            Ref. [#marinescu]_.
            Args:
                l (int): orbital angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: effective charge (in a.u.)
         """
        l = ch.l
         
        return 1.0 + (self.Z - 1) * np.exp(-self.a1[l] * r) - \
            r * (self.a3[l] + self.a4[l] * r) * np.exp(-self.a2[l] * r)
    
    def potential(self, ch, r):
        """ returns total potential that electron feels
            Total potential = core potential + Spin-Orbit interaction
            Args:
                l (int): orbital angular momentum
                s (float): spin angular momentum
                j (float): total angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: potential (in a.u.)
        """
        

        l = ch.l
        j = ch.j
        s = ch.s
        
        so_factor = 0 if self.include_so else 1
        
        if l < 4 and self.use_model:
            return self.core_potential(ch, r) + so_factor*cs.fine_structure**2 / (2.0 * r**3) * \
                (j * (j + 1.0) - l * (l + 1.0) - s * (s + 1)) / 2.0
        else:
            # act as if it is a Hydrogen atom
            return -1. / r + so_factor*cs.fine_structure**2 / (2.0 * r**3) * \
                (j * (j + 1.0) - l * (l + 1.0) - s * (s + 1)) / 2.0

# Simple instance of model_potential for pure 1/r coulomb potential (not clear that this is used)
coulomb_potential = model_potential(0,[0]*4,[0]*4,[0]*4,[0]*4,[1e-3]*4,1,include_so = True)
          
class core_state:
    """ Class to represent core state used in MQDT formulation.
    
    The core state represents the quantum numbers of the inner electrons (s,l,j) and nucleus (i), adding up to a total angular momentum f of the core.
    There is no mF because a complete state is not formed until this adds with the Rydberg electron.

    There is also an associated ionization energy Ei_Hz, which is used to calculate the outer electron wavefunction associated with each channel.

    Lastly, the dipole and quadrupole polarizabilities alpha_d_a03 and alpha_q_a05 are included to calculate the polarization contribution to the quantum
    defect for high-l states.
    """
    
    def __init__(self,qn,Ei_Hz,tt='sljif',config='',potential = None, alpha_d_a03 = 0, alpha_q_a05 = 0):
        """ Example:
            
            171Yb Fc=0
            core_state((1/2,0,1/2,1/2,0), Ei = 50044.123, tt = 'sljif', config = '6s1/2 Fc=0')
            
        """
        
        if tt == 'sljif':
            
            self.s = qn[0]
            self.l = qn[1]
            self.j = qn[2]
            self.i = qn[3]
            self.f = qn[4]
            
        else:
            print("tt ", tt, " not currently supported.")
        
        self.Ei_Hz = Ei_Hz
        self.Ei_au = self.Ei_Hz/(2*cs.Rydberg*cs.c)
        self.tt = tt
        self.config = config
        
        self.potential = potential

        self.alpha_d_a03 = alpha_d_a03
        self.alpha_q_a05 = alpha_q_a05
        
    def __repr__(self):
        return self.config
    
    def __eq__(self,other):
        if self.s == other.s and self.l == other.l and self.j == other.j and self.i == other.i and self.f == other.f and self.Ei_Hz == other.Ei_Hz and self.config == other.config:
            return True
        else:
            return False
        
    
class channel:
    
    def __init__(self,core,qn,tt = 'sljf',defect = None, no_me = False):
        # FIXME: defect is not used anymore and should be removed\
        """
        An MQDT channel consists of a core state and a Rydberg electron state. The core state is an instance of core_state,
        while the Rydberg electron state is specified by the quantum numbers (s,l,j) of the electron.

        Example:
            
            171Yb F=1/2 S channels
            coreFc0 = core_state((1/2,0,1/2,1/2,0), Ei = 50044.123, tt = 'sljif', config = '6s1/2 Fc=0')
            channel(coreFc2,(1/2,0,1/2), tt = 'slj')
            
            coreFc1 = core_state((1/2,0,1/2,1/2,1), Ei = 50044.123 + delta, tt = 'sljif', config = '6s1/2 Fc=1')
            channel(coreFc1,(1/2,0,1/2), tt = 'slj')
        
            """
        
        self.core = core
        self.tt = tt
        self.qn = qn
        self.no_me = no_me
        
        if defect is None:
            self.defects = []
        elif type(defect) is list:
            self.defects = defect
        else:
            self.defects = [defect]
        
        if tt == 'sljf':
            # it is not clear why this is here -- it doesn't seem like we should ever be asking about the quantum number f of the Rydberg electron
            # because there is explicitly no nuclear contribution to the Rydberg electron (it's in the core). It appears that this is a holdover,
            # and all of the actual instances of models use 'slj'.
            self.s = qn[0]
            self.l = qn[1]
            self.j = qn[2]
            self.f = qn[3]
            
        elif tt == 'slj':
            self.s = qn[0]
            self.l = qn[1]
            self.j = qn[2]
            self.f = self.j # total angular momentum of Rydberg electron
            
        else:
            print("tt ", tt, " not currently supported in channel.__init__().")
            return None
            
    # def get_defect(self,n):
        
    #     if len(self.defects)==0:
    #         return 0
        
    #     for d in self.defects:
    #         if d.is_valid(n):
    #             return d.get_defect(self,n)
            
    def __repr__(self):
        if len(self.defects) == 0:
            defstr="(no defect defined)"
        else:
           defstr= "(with defect model)"
        return "Channel, t=%s, qn=%s, core: %s" % (self.tt,repr(self.qn),self.core.config)
    
    def __eq__(self,other):
        
        if self.core == other.core and self.s == other.s and self.l == other.l and self.j == other.j and self.f == other.f:
            return True
        return False
            
class state_mqdt:
    
    def __init__(self,atom,qn,Ai,Aalpha,channels,energy_Hz,tt='npfm',eq_cutoff_Hz = 20000):
        """
        Initializes an MQDT state with specified parameters.

        Args:
            atom (AlkalineAtom): The atom for which the MQDT state is defined.
            qn (tuple): Quantum numbers for the state.
            Ai (float): Ionization energy in Hz.
            Aalpha (float): Polarizability.
            channels (list): List of channels associated with the state.
            energy_Hz (float): Energy of the state in Hz.
            tt (str, optional): Type tag for the quantum numbers. Defaults to 'npfm'.
            eq_cutoff_Hz (float, optional): Energy cutoff in Hz for equality checks. Defaults to 20000 Hz.
        """
        
        self.atom = atom
        self.qn = qn
        self.tt = tt
        
        # notes: the quantum number 't' was from a brief attempt to introduce an extra quantum number to distinguish series with the same quantum number (ie, S F=1/2 series in 171Yb).
        # however, we realized that ordering was impossible so have given up, and it should not be used but is left here so it doesn't break anything else...
        if tt=='npfm':
            self.n = qn[0]
            self.parity = qn[1]
            self.f = qn[2]
            self.t = 0
            self.m = qn[3]
            
        elif tt=='npftm':
            self.n = qn[0]
            self.parity = qn[1]
            self.f = qn[2]
            self.t = qn[3]
            self.m = qn[4]

        elif tt=='vpfm':
            self.n = qn[0]
            self.v = qn[0]
            self.parity = qn[1]
            self.f = qn[2]
            self.m = qn[3]
            
        elif tt=='nsljfm':
            
            self.n = qn[0]
            self.s = qn[1]
            self.parity = (-1)**qn[2]
            self.l = qn[2]
            self.j = qn[3]
            self.f = qn[4]
            self.m = qn[5]
            
        else:
            print("tt=",tt," not supported by state_mqdt.")
        
        self.channels = channels
        self.Ai = Ai
        self.Aalpha = Aalpha
        
        self.energy_Hz = energy_Hz
        
        self.hashval = id(self)

        self.eq_cutoff_Hz = eq_cutoff_Hz
        
    def get_channel_idx(self,ch):
        for idx,c in enumerate(self.channels):
            if c == ch:
                return idx
        return None
        
    def get_energy_Hz(self):
        return self.energy_Hz
    
    def get_energy_au(self):
        """ Get state energy in atomic units (Hartree) """
        return self.energy_Hz/(2*cs.Rydberg*cs.c)
    
    def get_g(self):
        return self.atom.get_g(self)
    
    def __repr__(self):
        # print a nice ket
        return self.atom.repr_state(self)

    def __eq__(self,other):
        # check for equality
        
        # it is about 3x faster to do this with the hash, since we've already computed it,
        # even though it's not as clear
        if self.hashval == other.hashval:
            return True
        else:
        
            if self.atom == other.atom and np.round(self.nub*10000)==np.round(other.nub*10000):#np.abs(self.energy_Hz - other.energy_Hz)<self.eq_cutoff_Hz:
                if self.qn == other.qn:
                    return True
                
                # dont' include tau in comparison
                if self.tt == 'npftm' and other.tt == 'npftm' and np.all([self.qn[x] == other.qn[x] for x in (0,1,2,4)]):
                    return True
        
        return False
    
    def __hash__(self):
        #return self.atom.__hash__() + hash(tuple(self.qn.values())+(self.tt,))
        return self.hashval
        
    def __ne__(self,other):
        return not self.__eq__(other)
        
    def allowed_multipole(self,other,k=1,qIn=None):
        if self.atom==other.atom:
            return self.atom.allowed_multipole(self,other,k=k,qIn=qIn)
        else:
            return False

    def get_multipole_me(self,other,k=1,qIn=None,operator=None):
        """ get multipole matrix element between these states. polarization is implicit.
        argument is final state, self is initial state.
        qIn is optional argument restricting polarization (will return 0 otherwise)
        Answer is in atomic units, e*a0 """
        
        # we have typically already checked this
        # if not self.allowed_multipole(other,k=k,qIn=qIn):
        #     return 0
        
        return self.atom.get_multipole_me(self,other,k=k,qIn=qIn,operator=operator)
        
        
class state(state_mqdt):
    
    """ Internal state representation designed to handle MQDT.
    State is expressed as a linear combination of channels:
        
        \sum_i A_i ch_i
        
    """
    
    def __init__(self,atom,qn,channel,energy_Hz,tt='npfm'):
        """ Default constructor for single-channel state """
        
        # if tt=='npfm':
        #     n = qn[0]
        #     if energy_Hz is None:
        #         self.energy_Hz = atom.get_energy_Hz_from_defect(n,channel.get_defect(n))
    
        return super().__init__(atom,qn,[1.0],[1.0],[channel],energy_Hz,tt=tt)


class single_basis:
    """ Class to define a basis of single-atom states, with helper functions to construct and manipulate Hamiltonians. """
    
    def __init__(self):
        """ Constructor for single_basis. """
        self.states = []
        self.highlight = []
        
    def add(self,st):
        
        if len(self.states) == 0:
            self.states.append(st)
            return
        
        # should only add states of the same atom type and term type into a basis
        if self.find(st) is None and st.atom==self.states[0].atom:
            self.states.append(st)
        else:
            pass
            # print("Error--duplicate state ",repr(st))
            # print("Conflicts with ", self.states[self.find(st)])
    
    def dim(self):
        return len(self.states)
    
    def find(self,st):
        match = -1
        
        for ii in range(len(self.states)):
            if self.states[ii] == st:
                return ii
    
        return None
    
    def getClosest(self,ev):
        """ return the state from the basis that has maximum overlap with the given eigenvector """
        idx = np.argmax(np.abs(ev)**2)
        return self.states[idx]
    
    def getVec(self,s):
        """ get vector corresponding to state s """
        idx = self.find(s)
        
        vec = np.zeros(self.dim())
        vec[idx] = 1
        return vec
    
    # def fill(self,s0,include_opts={}):
    #     """ fill in states in a range around the initial state s0.
    #     dipole_allowed option specifies that only states that are E1 allowed
    #     should be included.
    #     """
        
    #     # see atom.get_nearby for discussion about options and where states are included/excluded
        
    #     # potential Alkaline FIXME: iterate over multiple spin states
    #     # to allow for singlet-triplet mixing
        
    #     self.s0 = s0
    #     self.add(s0)
        
    #     # this is a default set of options that can get overriden/added to
    #     # by putting things in include_opts. It is deisgned to be tt-agnostic
    #     self.opts = {'dn': 2, 'dl': 2, 'dm': 1,'dipole_allowed':False, 'include_fn': lambda s,s0: True}
        
    #     for k,v in include_opts.items():
    #         self.opts[k] = v
        
    #     qn_list = s0.atom.get_nearby(self.s0,include_opts = self.opts)
        
    #     for newqn in qn_list:
                            
    #         #newst = s0.atom.get_state(newqn, tt = s0.tt)
    #         newst = s0.atom.get_state(newqn)
            
    #         if newst is not None:
    #             dipole_ok = s0.allowed_multipole(newst,k=1)
                
    #             fn_ok = self.opts['include_fn'](newst,s0)
                
    #             if (fn_ok) and (dipole_ok or (not self.opts['dipole_allowed'])):
    #                 self.add(newst)

                                
    #     #self.highlight = [[s0,'n',None]]
    #     self.add_highlight(s0,'n')

    def fill(self, s0, include_opts={}):
        """
        Fills the basis with states that are nearby the initial state `s0` based on specified options.

        This method populates the basis with states that are close to the given state `s0` according to the quantum
        numbers and options provided. It allows for the inclusion of states based on dipole transitions and other
        criteria defined in `include_opts`.

        Parameters:
            s0 (State): The initial state around which nearby states will be filled.
            include_opts (dict): Options to specify which states to include. Options include:
                                 - dn: Change in principal quantum number (default: 2)
                                 - dl: Change in orbital quantum number (default: 2)
                                 - dm: Change in magnetic quantum number (default: 1)
                                 - dipole_allowed (bool): If True, only include states that are dipole-allowed transitions from `s0` (default: False)
                                 - include_fn (function): A function that takes a state and the initial state `s0`, returning True if the state should be included.

        Returns:
            None: This method modifies the basis in-place by adding states.
        """

        self.s0 = s0
        self.add(s0)

        # this is a default set of options that can get overriden/added to
        # by putting things in include_opts. It is deisgned to be tt-agnostic
        self.opts = {'dn': 2, 'dl': 2, 'dm': 1, 'dipole_allowed': False, 'include_fn': lambda s, s0: True}

        for k, v in include_opts.items():
            self.opts[k] = v

        st_list = s0.atom.get_nearby(self.s0, include_opts=self.opts)

        for newst in st_list:

            if newst is not None:
                dipole_ok = s0.allowed_multipole(newst, k=1)

                fn_ok = self.opts['include_fn'](newst, s0)

                if (fn_ok) and (dipole_ok or (not self.opts['dipole_allowed'])):
                    self.add(newst)

        self.add_highlight(s0, 'n')
        
    def add_highlight(self,s1,sym='n',s2=None):
        # add an entry to highlighted pair list
        
        if sym=='n' and type(s2) is type(None):
            self.highlight.append((s1,'n',None,self.getVec(s1)))
            
        if type(s2) is type(None):
            return # shouldn't get here
        
        if sym == 'g':
            vec = (self.getVec(s1) + self.getVec(s2))/np.sqrt(2)
        elif sym == 'u':
             vec = (self.getVec(s1) - self.getVec(s2))/np.sqrt(2)
             
        self.highlight.append((s1,sym,s2,vec))
    
    def compute_hamiltonian(self, ponderomotive=None):
        """
        Compute the Hamiltonian matrices for the system:

        - HEz: The electric field Hamiltonian.
        - HBz: The magnetic field Hamiltonian.
        - HBdiam: The diamagnetic interaction.
        - H0: The zero-field Hamiltonian matrix (ie, state energies).
        - Hpond: The ponderomotive potential Hamiltonian (if given).

        They are stored separately, so the total Hamiltonian can be efficiently computed by multiplying these by the appropriate field strengths and summing.

        Parameters:
            ponderomotive (callable, optional): A function that takes two states and returns the ponderomotive interaction matrix element between them. If None, the ponderomotive interaction is not included in the Hamiltonian.

        Modifies:
            self.HEz: Updates the electric field interaction matrix.
            self.HBz: Updates the magnetic field interaction matrix.
            self.HBdiam: Updates the diamagnetic interaction matrix.
            self.H0: Updates the zero-field Hamiltonian matrix.
            self.Hpond: Updates the ponderomotive Hamiltonian matrix if `ponderomotive` is not None.

        Returns:
            None
        """
        Nb = self.dim()

        self.HEz = np.zeros((Nb,Nb))
        self.HBz = np.zeros((Nb,Nb))
        self.HBdiam = np.zeros((Nb,Nb))
        self.H0 = np.zeros((Nb,Nb))
        self.Hpond = np.zeros((Nb,Nb))
            
        # first, compute H0
        for ii in range(Nb):
            self.H0[ii,ii] = self.states[ii].get_energy_Hz() - self.s0.get_energy_Hz()
            
        # if we are not using an MQDT model, HBz is also diagonal so we can get that out of the way here:
        if not isinstance(self.s0,state_mqdt):
            for ii in range(Nb):
                self.HBz[ii,ii] = self.states[ii].get_g()*self.states[ii].m

        #ii (row idx) is final state
        #jj (col idx) is initial state
        for ii in range(Nb):
            for jj in range(ii,Nb): # only do the upper right triangle + diagonal (for ponderomotive)
                
                sii = self.states[ii]
                sjj = self.states[jj]
                
                q = sjj.m - sii.m

                # At one point, we determined it was faster to check whether it was allowed before attempting to compute.
                # Since we are only considering z-oriented E-fields, qIn = [0]    
                if sii.allowed_multipole(sjj,k=1,qIn=[0]):
                    #self.HEz[ii,jj] += sii.get_dipole_me(sjj,qIn=[0])
                    self.HEz[ii,jj] += sii.get_multipole_me(sjj,qIn=(0,),k=1)
                    
                if ponderomotive is not None:
                    self.Hpond[ii,jj] += ponderomotive.get_me(sii,sjj)
                    
                if isinstance(self.s0,state_mqdt):
                    self.HBz[ii,jj] += sii.atom.get_magnetic_me(sii,sjj)
                    self.HBdiam[ii,jj] = sii.atom.diamagnetic_int(sii,sjj)


        # since we only did upper-right triangle, add transposed version
        self.HEz = self.HEz + np.conjugate(np.transpose(self.HEz)) # this has no diagonal elements
        self.Hpond = self.Hpond + np.conjugate(np.transpose(self.Hpond)) - np.diagflat(np.diagonal(self.Hpond))
        self.HBz = self.HBz + np.conjugate(np.transpose(self.HBz)) - np.diagflat(np.diagonal(self.HBz))
        self.HBdiam = self.HBdiam + np.conjugate(np.transpose(self.HBdiam)) - np.diagflat(np.diagonal(self.HBdiam))
        
    def compute_energies(self,env):
        """
        Compute the total energies and eigenstates for the system under specified environmental conditions.

        Args:
            env (Environment): An object containing the environmental parameters like electric field strength, magnetic field strength, and laser intensity for ponderomotive potential.
            diam (bool): A flag to include diamagnetic contributions in the Hamiltonian if set to True (default: False).

        Returns:
            np.ndarray: An array containing tuples of (energy, overlap with target state, index of the state) for highlighted state pairs.

        Also updates internal variables self.es and self.ev with the computed eigenvalues and eigenvectors.
        """
        
        Nb = self.dim()
        
        self.Htot = np.zeros((Nb,Nb),dtype='float64')
        
        ub = cs.physical_constants['Bohr magneton in Hz/T'][0]*1e-4 # in Hz/Gauss
        self.Htot += self.H0 + env.Ez_Vcm*100*cs.e*a0*self.HEz/cs.h + ub*env.Bz_Gauss*self.HBz + env.Intensity_Wm2*self.Hpond/cs.h
        if env.diamagnetism == True:
            self.Htot += (env.Bz_Gauss/10000)**2*self.HBdiam/cs.h
        
        self.es,self.ev = np.linalg.eigh(self.Htot)
        
        # eig() does not sort eigenvalues, so do that first
        sort_idx = np.argsort(self.es)
        self.es = self.es[sort_idx]
        self.ev = self.ev[:,sort_idx]
        
        # now find highlighted pairs, and return (energy,overlap)
        
        ret = []
        for s1,sym,s2,targetState in self.highlight:

            targetState = targetState / np.linalg.norm(targetState)
            #print(targetState)
            ov = [np.abs(np.sum(targetState*self.ev[:,ii]))**2 for ii in range(Nb)]
            idx = np.argmax(ov)
            
            ret.append([self.es[idx],ov[idx],idx])

        return np.array(ret)
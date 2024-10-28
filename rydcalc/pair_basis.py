import scipy as sp

from rydcalc import *

class pair:
    
    def __init__(self,s1,s2):
        self.s1 = s1
        self.s2 = s2
        
        self.energy_Hz = s1.get_energy_Hz() + s2.get_energy_Hz()
        
        self.flag = ''
    
    def __repr__(self):
        # print a nice ket
        return repr(self.s1) + " " + repr(self.s2) + " " + self.flag

    def __eq__(self,other):
        # check for equality
        
        if self.s1 == other.s1 and self.s2 == other.s2:
            return True
        else:
            return False
    
    def __ne__(self,other):
        return not self.__eq__(other)
                            

class pair_basis:
    class pair_basis:
        """
        A class to manage a basis of pair states for calculations involving interactions between pairs of states.
        """
    
    def __init__(self):
        self.pairs = []
        self.highlight = []
    
    def add(self,p):
        if self.find(p) is None:
            self.pairs.append(p)
        else:
            pass
            #print("Error--duplicate pair ",p)
    
    def dim(self):
        return len(self.pairs)
    
    def find(self,p):
        match = -1
        
        for ii in range(len(self.pairs)):
            if self.pairs[ii] == p:
                return ii
    
        return None
    
    def getClosest(self,ev):
        """ return the state that has maximum overlap with the given eigenvalue """
        idx = np.argmax(np.abs(ev)**2)
        return self.pairs[idx]
    
    def getVec(self,p):
        """ get vector corresponding to state p """
        idx = self.find(p)
        
        vec = np.zeros(self.dim())
        vec[idx] = 1
        return vec

    def fill(self, p0, include_opts={}, dm_tot=[-2, -1, 0, 1, 2]):
        """
        Fills the pair basis for the given pair state `p0` with options specified in `include_opts`.

        This method initializes the pair basis states considering the options provided for dipole transitions,
        exchange interactions, and other specified conditions. It populates the basis with all possible state
        combinations from single_basis instances `sb1` and `sb2`, which are filled based on the states `s1` and `s2`
        of `p0`.

        Parameters:
            p0 (PairState): The pair state for which the basis is to be filled.
            include_opts (dict): Options to include specific conditions while filling the basis.
                Keys can include 'dn', 'dl', 'dm', 'dipole_allowed', 'force_exchange', and 'pair_include_fn'.
                - 'force_exchange' forces the inclusion of pairs with states swapped
                - 'pair_include_fn' is a lambda function taking two pairs and returning True/False (ie, lambda p,p0: np.abs(p.energy_Hz-p0.energy_Hz) < 1e9)
            dm_tot (list of int): List of allowed changes in the total magnetic quantum number (delta m).

        Notes:
            - The method checks for dipole allowed transitions if 'dipole_allowed' is True.
            - Exchange interactions are considered if the states belong to the same atom and other conditions
              specified in `force_exchange` or delta m conditions are met.
            - The method also handles highlighting of states if exchange is considered.
        """
        """ fill in basis of states that are needed to calculate interactions of p0.

        dipole_allowed option specifies that only states that are e1 allowed from initial
        states should be included. Note that this restriction is not acceptable for circular
        states in electric fields, where the linear DC stark shift between other states
        must be taken into account.

        If dipole_allowed is True, dm_tot specifies which total deltaM should be included.
        This is useful to restrict computation basis for certain angles, ie, when th=0
        only dM = 0 is relevant.
        """

        self.p0 = p0

        self.add(p0)

        self.opts = {'dn': 2, 'dl': 2, 'dm': 1, 'dipole_allowed': True, 'force_exchange': False, 'pair_include_fn': lambda p, p0: True}

        for k, v in include_opts.items():
            self.opts[k] = v

        # if we are dealing with different states where exchange is allowed,
        # ofen need to add flipped state explicitly
        if p0.s1.atom == p0.s2.atom:  # and p0.s1.tt == p0.s2.tt:
            # if the two states are the same atomic species and specified with the same
            # kind of term, then we will want to consider processes where (s1,s2) -> (s2,s1)
            # if that transition is allowed (ie, if deltaM is less than two for second-order
            # E1 transitions). If we add quadrupole, etc., will need to relax this constraint
            if (np.abs(p0.s1.m - p0.s2.m) <= 2 or self.opts['force_exchange']) and p0.s1 != p0.s2:
                self.consider_exchange = True
            else:
                self.consider_exchange = False
        else:
            # if the two states belong to different atoms or have different term symbols,
            # then we do not consider exchange. This would be the case for inter-species
            # interactions (ie, Rb-Cs) or low-L - circular interactions, if we specify
            # the circular states using 'nlm' terms to use analytic wavefunctions
            self.consider_exchange = False

        self.sb1 = single_basis()
        self.sb1.fill(p0.s1, include_opts=self.opts)

        self.sb2 = single_basis()
        self.sb2.fill(p0.s2, include_opts=self.opts)

        for s1 in self.sb1.states:
            for s2 in self.sb2.states:
                # if self.sb1.find(s2) is not None:
                #     s2_temp = self.sb1.states[self.sb1.find(s2)]
                # else:
                #     s2_temp = s2

                if self.opts['dipole_allowed']:
                    if not (p0.s1.allowed_multipole(s1, k=1) and p0.s2.allowed_multipole(s2, k=1)):
                        continue

                # implement restriction on change in total m
                delta_m_tot = s1.m + s2.m - (p0.s1.m + p0.s2.m)
                if self.opts['dipole_allowed'] and not (delta_m_tot in dm_tot):
                    continue

                if not self.opts['pair_include_fn'](pair(s1, s2), self.p0):
                    continue

                self.add(pair(s1, s2))
                if self.consider_exchange:
                    # if we allow exchange interactions, explicitly add reversed
                    # states to the Hamiltonian. This significantly increases
                    # basis size
                    self.add(pair(s2, s1))

        # highlight is a list of tuples, specifing (p1,'g'/'u'/'n',p2,targetstate)
        if self.consider_exchange:  # p0.s1 == p0.s2 or p0.s1.atom != p0.s2.atom:
            # self.highlight = [(p0,'g',pair(p0.s2,p0.s1)),(p0,'u',pair(p0.s2,p0.s1))]
            self.addHighlight(p0, 'g', pair(p0.s2, p0.s1))
            self.addHighlight(p0, 'u', pair(p0.s2, p0.s1))
        else:
            self.addHighlight(p0, 'n')
            # self.highlight = [(p0,'n',None,self.getVec(p1))]
            
    
    def addHighlight(self,p1,sym='n',p2=None):
        """
        Adds a highlighted pair to the highlight list with a specified symmetry.

        Args:
            p1 (Pair): The primary pair involved in the highlight.
            sym (str): The symmetry type of the highlight ('g', 'u', or 'n').
            p2 (Pair, optional): The secondary pair involved in the highlight. Defaults to None.

        Notes:
            - If `sym` is 'n' and `p2` is None, the highlight is added with no secondary pair and a vector derived from `p1`.
            - If `sym` is 'g' or 'u', the highlight is added with both pairs and a vector calculated as a symmetric or antisymmetric combination of the vectors from `p1` and `p2`.
        """
        
        if sym=='n' and type(p2) is type(None):
            self.highlight.append((p1,'n',None,self.getVec(p1)))
            
        if type(p2) is type(None):
            return # shouldn't get here
        
        if sym == 'g':
            vec = (self.getVec(p1) + self.getVec(p2))/np.sqrt(2)
        elif sym == 'u':
             vec = (self.getVec(p1) - self.getVec(p2))/np.sqrt(2)
             
        self.highlight.append((p1,sym,p2,vec))
        

    def computeHamiltonians(self, multipoles=[[1,1]]):
        """
        Compute the Hamiltonians for the system considering the specified multipoles.

        This function initializes and computes the Hamiltonians for electric fields, magnetic fields, diamagnetic interaction, and the multipole interaction.
        
        All hamiltonians (including the different q values for the multipole interaction) are stored separately, so they can be resummed with appropriate
        coefficients to rapidly compute eigenvalues at different field strengths, distances and orientations.

        Args:
            multipoles (list of list of int): A list of pairs specifying the orders of the multipoles to include.
                Each pair is of the form [k1, k2] where k1 and k2 are the orders of the multipoles on each atom. When including [1,2], also need to include [2,1]

        """
        
        self.multipoles = multipoles

        Nb = self.dim()

        self.HEz = np.zeros((Nb,Nb))
        self.HBz = np.zeros((Nb,Nb))
        self.HBdiam = np.zeros((Nb,Nb))
        self.H0 = np.zeros((Nb,Nb))
        
        # HInt is 5xNxN, where the first index is the total change
        # in magnetic quantum number. Keeping track of it this way allows
        # the angular dependence to be easily worked out later
        #self.HInt = np.zeros((5,Nb,Nb))
        
        self.HInt = []
        
        for mm in self.multipoles:
            # need to have an Nb x Nb matrix for each possible deltaMtotal
            dm_max = mm[0] + mm[1]
            self.HInt.append(np.zeros((2*dm_max+1,Nb,Nb)))
            

        # first, compue H0 and HBz, which are diagonal
        for ii in range(Nb):
            self.H0[ii,ii] = self.pairs[ii].energy_Hz - self.p0.energy_Hz
            #self.HBz[ii,ii] = self.pairs[ii].s1.get_g()*self.pairs[ii].s1.m + self.pairs[ii].s2.get_g()*self.pairs[ii].s2.m

        self._compute_HBz()
        self._compute_HBdiam()
        self._compute_HEz_Hint_fast()
        
    def _compute_HBz(self):
        """ Compute the Zeeman Hamiltonian. """
        
        # if we don't have an MQDT model, can just enter diagonal matrix elements
        if (not isinstance(self.pairs[0].s1,state_mqdt)) or (not isinstance(self.pairs[0].s2,state_mqdt)):
            for ii in range(self.dim()):
                self.HBz[ii,ii] = self.pairs[ii].s1.get_g()*self.pairs[ii].s1.m + self.pairs[ii].s2.get_g()*self.pairs[ii].s2.m
                
            return
    
        # if we do, put in off-diagonal matrix elements as well
        # NB: this might fail for interactions between MQDT and non-MQDT, because we don't have get_magnetic_me defined for non-MQDT states.
        for ii in range(self.dim()):
            for jj in range(self.dim()):
                
                pii = self.pairs[ii]
                pjj = self.pairs[jj]
                
                if pii.s2 == pjj.s2:
                    self.HBz[ii,jj] += pii.s1.atom.get_magnetic_me(self.pairs[ii].s1,self.pairs[jj].s1)
                    
                if pii.s1 == pjj.s1:
                    self.HBz[ii,jj] += pii.s2.atom.get_magnetic_me(self.pairs[ii].s2,self.pairs[jj].s2)
        
        return

    def _compute_HBdiam(self):
        """ Compute the Diamagnetic Hamiltonian. """

        # if we do, put in off-diagonal matrix elements as well
        # NB: this might fail for interactions between MQDT and non-MQDT, because we don't have get_magnetic_me defined for non-MQDT states.
        for ii in range(self.dim()):
            for jj in range(self.dim()):

                pii = self.pairs[ii]
                pjj = self.pairs[jj]

                if pii.s2 == pjj.s2:
                    self.HBdiam[ii, jj] += pii.s1.atom.diamagnetic_int(self.pairs[ii].s1, self.pairs[jj].s1)

                if pii.s1 == pjj.s1:
                    self.HBdiam[ii, jj] += pii.s2.atom.diamagnetic_int(self.pairs[ii].s2, self.pairs[jj].s2)

        return



    def _compute_HEz_Hint(self):
        printDebug = False
        computeHInt = True

        #ii (row idx) is final state
        #jj (col idx) is initial state
        for ii in range(self.dim()):
            for jj in range(ii,self.dim()): # only do the upper right triangle + diagonal
                
                pii = self.pairs[ii]
                pjj = self.pairs[jj]
                
                #print(pii,pjj)
                
                # if pairs are connected by a single-atom E1 transition with q=0,
                # add it to HEz

                # NB: putting the == first keeps this from evaluating the second part
                # when it's not needed, which saves a lot of time
                if pii.s2 == pjj.s2 and pii.s1.allowed_multipole(pjj.s1,k=1,qIn=(0,)):
                #if pii.s1.allowed_e1(pjj.s1,qIn=[0]) and pii.s2 == pjj.s2:
                    #self.HEz[ii,jj] += pii.s1.get_dipole_me(pjj.s1,qIn=[0])
                    self.HEz[ii,jj] += pii.s1.get_multipole_me(pjj.s1,qIn=(0,),k=1)
                 
                if pii.s1 == pjj.s1 and pii.s2.allowed_multipole(pjj.s2,k=1,qIn=(0,)):    
                #if pii.s2.allowed_e1(pjj.s2,qIn=[0]) and pii.s1 == pjj.s1:
                    #self.HEz[ii,jj] += pii.s2.get_dipole_me(pjj.s2,qIn=[0])
                    self.HEz[ii,jj] += pii.s2.get_multipole_me(pjj.s2,qIn=(0,),k=1)

                if self.HEz[ii,jj] != 0 and printDebug:
                    print(pii, "<-(Ez)-> ", pjj)
                    
                # if pairs are connected by allowed E1 transitions in both atoms, add to Hint:

                if computeHInt:
                    
                    for mm,HInt in zip(self.multipoles, self.HInt):
                        
                        # first check that the pair states reflect allowed transitions for given multipole
                        #if pii.s1.allowed_e1(pjj.s1) and pii.s2.allowed_e1(pjj.s2):
                            
                        if pii.s1.allowed_multipole(pjj.s1,k=mm[0]) and pii.s2.allowed_multipole(pjj.s2,k=mm[1]):
                            
                            # q = final-initial, ie ii-jj
                            q1 = -(pii.s1.m - pjj.s1.m)
                            q2 = -(pii.s2.m - pjj.s2.m)
        
                            qtot = int(q1+q2)
                            qidx = qtot + (mm[0] + mm[1]) #index for HInt arr.
                            
                            d1 = pii.s1.get_multipole_me(pjj.s1,k=mm[0])
                            d2 = pii.s2.get_multipole_me(pjj.s2,k=mm[1])
        
                            #cg = CG(mm[0],q1,mm[1],q2,mm[0]+mm[1],q1+q2).doit().evalf()
                            cg = CG(mm[0],q1,mm[1],q2,mm[0]+mm[1],q1+q2)
        
                            me = cg*d1*d2
        
                            if me != 0:
                            # will multiply later by other factors which are not state-dependent
                                HInt[qidx,ii,jj] = me
        
                            if me != 0  and printDebug:
                                print(pii, "<-(",qtot,mm,")-> ", pjj)
        
        # since we only did upper-right triangle, add transposed version
        self.HEz = self.HEz + np.conjugate(np.transpose(self.HEz))
        
        for mm,HInt in zip(self.multipoles, self.HInt): 
            for qidx in range(len(HInt)):
                HInt[qidx] = HInt[qidx] + np.conjugate(np.transpose(HInt[qidx])) - np.diagflat(np.diagonal(HInt[qidx]))
    
    def _compute_HEz_Hint_fast(self):
        """ This fast version of computing HEz and Hint works by iterating over the pair basis
        in blocks grouped by the first state, s1. For each block, we check if the required
        s1 transition is allowed (ie, dipole-dipole in the case we are considering [1,1] multipole).
        
        If it is not, we skip the entire block (this is the main time saving)
        
        If it is, we compute the upper diagonal of cthe block, as before.
        
        Note that with caching enabled, the main thing that we are trying to minimize
        is calls to allowed_multipole().
        """
        
        printDebug = False
        computeHInt = True
        
        # generate a list of all unique states s1
        unique_s1 = list(set([p.s1 for p in self.pairs]))
        
        # for each state, make a list of the pair indexes where they occur
        unique_s1_pair_idx = []

        ttot=0

        for us1 in unique_s1:
            indices = [en[0] for en in filter(lambda p: p[1].s1==us1, enumerate(self.pairs))]
            unique_s1_pair_idx.append(indices)


        # now we loop over blocks of pairs with same s1
        for s1_i,pair_idx_i in zip(unique_s1,unique_s1_pair_idx):
            
            for s1_j,pair_idx_j in zip(unique_s1,unique_s1_pair_idx):

                # Ez
                # compute Ez if s1_i == s1_j or if there is a ME for s1_i->j
                if s1_i == s1_j or s1_i.allowed_multipole(s1_j,k=1,qIn=(0,)):
                    
                    s1_same = True if s1_i == s1_j else False

                    # loop over upper diagonal
                    
                    for ii in pair_idx_i:
                        for jj in filter(lambda x: x>= ii, pair_idx_j):
                            
                            pii = self.pairs[ii]
                            pjj = self.pairs[jj]
                            
                            if (not s1_same) and pii.s2 == pjj.s2:
                                self.HEz[ii,jj] += pii.s1.get_multipole_me(pjj.s1,qIn=(0,),k=1)
                             
                            if s1_same and pii.s2.allowed_multipole(pjj.s2,k=1,qIn=(0,)):
                                self.HEz[ii,jj] += pii.s2.get_multipole_me(pjj.s2,qIn=(0,),k=1)


                # now look at interactions

                for mm,HInt in zip(self.multipoles, self.HInt):
                    
                    # if there is the right transition allowed on s1
                    
                    if s1_i.allowed_multipole(s1_j,k=mm[0]):
                        
                        for ii in pair_idx_i:
                            for jj in filter(lambda x: x>= ii, pair_idx_j):
                                
                                pii = self.pairs[ii]
                                pjj = self.pairs[jj]
                                
                                # already checked s1
                                #if pii.s1.allowed_multipole(pjj.s1,k=mm[0]) and pii.s2.allowed_multipole(pjj.s2,k=mm[1]):
                                if pii.s2.allowed_multipole(pjj.s2,k=mm[1]):
                                    
                                    # q = final-initial, ie ii-jj
                                    q1 = -(pii.s1.m - pjj.s1.m)
                                    q2 = -(pii.s2.m - pjj.s2.m)
                
                                    qtot = int(q1+q2)
                                    qidx = qtot + (mm[0] + mm[1]) #index for HInt arr.

                                    d1 = pii.s1.get_multipole_me(pjj.s1,k=mm[0])
                                    d2 = pii.s2.get_multipole_me(pjj.s2,k=mm[1])

                                    #cg = CG(mm[0],q1,mm[1],q2,mm[0]+mm[1],q1+q2).doit().evalf()
                                    cg = CG(mm[0],q1,mm[1],q2,mm[0]+mm[1],q1+q2)
                
                                    me = cg*d1*d2
                
                                    if me != 0:
                                    # will multiply later by other factors which are not state-dependent
                                        HInt[qidx,ii,jj] = me
                
                                    if me != 0  and printDebug:
                                        print(pii, "<-(",qtot,mm,")-> ", pjj)


        # since we only did upper-right triangle, add transposed version
        self.HEz = self.HEz + np.conjugate(np.transpose(self.HEz))
        
        for mm,HInt in zip(self.multipoles, self.HInt): 
            for qidx in range(len(HInt)):
                HInt[qidx] = HInt[qidx] + np.conjugate(np.transpose(HInt[qidx])) - np.diagflat(np.diagonal(HInt[qidx]))
                        
                
        
        
    def computeHtot(self,env,rum,th=np.pi/2,phi=0,interactions=True,multipoles=None):
        """ this computes the eigenvalues of the total hamiltonian for environment
        specified in env (E,B, etc.), for two atoms with relative axis (r,th,phi).
        
        Interactions can be turned off with option interactions.
        
        Returns energies/overlaps of eigenstates with maximum overlap on pairs in
        self.highlights. Also saves eigenstates/values for further analysis.
        
        If multipoles option is not specified, will use all avaialble hamiltonians from compute_hamiltonians.
        """
        Nb = self.dim()
        
        # save time if we can
        dtype = 'float64' if phi==0 else 'complex128'
        
        self.Htot = np.zeros((Nb,Nb),dtype=dtype)
        
        ub = cs.physical_constants['Bohr magneton in Hz/T'][0]*1e-4 # in Hz/Gauss
        self.Htot += self.H0 + env.Ez_Vcm*100*cs.e*a0*self.HEz/cs.h + ub*env.Bz_Gauss*self.HBz

        if env.diamagnetism == True:
            self.Htot+= self.HBdiam*(env.Bz_Gauss/10000)**2/cs.h
        
        if interactions:
            
            if multipoles is not None:
                for mm in multipoles:
                    if mm not in self.multipoles:
                        print("Warning--trying to compute interactions with %s multipole, but hamiltonian has only been computed for %s" % (repr(mm),repr(self.multipoles)))
                        
            
            self.HIntAll = np.zeros((Nb,Nb),dtype=dtype)
            
            for mm,HInt in zip(self.multipoles, self.HInt):
                
                if multipoles is not None and not(mm in multipoles):
                    continue
                
                dq_max = mm[0] + mm[1]
                for qidx in range(len(HInt)):
                    self.HIntAll += HInt[qidx]*self.interaction_prefactor(qidx-dq_max,rum,th,phi,mm=mm)
            
            self.Htot += self.HIntAll
        
        # NB: eigh is about 4x faster than eig
        self.es,self.ev = np.linalg.eigh(self.Htot)
        
        # eig() does not sort eigenvalues, so do that first
        sort_idx = np.argsort(self.es)
        self.es = self.es[sort_idx]
        self.ev = self.ev[:,sort_idx]
        
        # now find highlighted pairs, and return (energy,overlap)
        
        ret = []
        for p1,sym,p2,targetState in self.highlight:
            
            targetState = targetState / np.linalg.norm(targetState)

            ov = [np.abs(np.sum(targetState*self.ev[:,ii]))**2 for ii in range(Nb)]
            idx = np.argmax(ov)
            
            ret.append([self.es[idx],ov[idx],idx])

        return np.array(ret)
    
    
    def interaction_prefactor(self,qtot,r,th,phi,mm=[1,1]):
        """ returns coefficient to give interaction in Hz for r in microns """
        # recall that the scipy sph_harm function has both the order of l,m and th,phi reversed from Mathematica
        unit_scale = (cs.e)**2 * a0**(mm[0]+mm[1]) / (4*np.pi*cs.epsilon_0*cs.h)
        
        #return -unit_scale * np.sqrt(24*np.pi/5) * 1/r**(3) * np.conjugate(sp.special.sph_harm(qtot,2,phi,th))
        
        number_factor = (4*np.pi)**3 * np.math.factorial(2*mm[0] + 2*mm[1]) / ( np.math.factorial(2*mm[0]+1) * np.math.factorial(2*mm[1]+1) * (2*mm[0] + 2*mm[1] + 1) )
        number_factor *= (2*mm[0]+1)/(4*np.pi)*(2*mm[1]+1)/(4*np.pi)
        
        # number_factor values
        # Note that this is equal to sqrt term in Vaillant 2012 eq. 6, but multiplied by 4pi/(2k+1) because
        # our reduced matrix elements are of normalized spherical harmonics.
        #  [1,1]: 24*pi/5 
        #  [1,2]: 60*pi/7
        #  [2,2]: 280*pi/9
        #  [3,3]: 3696*pi/13
        
        ret = (-1)**mm[1] * unit_scale * np.sqrt(number_factor) * 1/(r*1e-6)**(mm[0]+mm[1]+1) * np.conjugate(sp.special.sph_harm(qtot,mm[0]+mm[1],phi,th))
    
        if phi==0:
            # in this case, we can save time by using real datatypes
            return np.real(ret)
        else:
            return ret
    
    def compute_Heff(self,subspace_pairs):
        """ compute the effective hamiltonian of the subspace_pairs to second order in the interaction Hamiltonian.
        Needs to be run after computeHtot """
        
        subspace_idx = [self.find(p) for p in subspace_pairs]
        
        # first get the restriction of Htot onto the subspace
        Heff_0 = self.Htot[np.ix_(subspace_idx,subspace_idx)]
        
        #N now sum over intermediate states
        for ii in range(self.dim()):
            if ii not in subspace_idx:
                
                for nn in range(len(subspace_idx)):
                    for mm in range(len(subspace_idx)):
                        dE_nn = self.pairs[subspace_idx[nn]].energy_Hz - self.pairs[ii].energy_Hz
                            
                        dE_mm = self.pairs[subspace_idx[mm]].energy_Hz - self.pairs[ii].energy_Hz

                        Heff_0[nn,mm] += 0.5*(self.HIntAll[subspace_idx[nn],ii]*self.HIntAll[ii,subspace_idx[mm]]/dE_nn + self.HIntAll[subspace_idx[mm],ii]*self.HIntAll[ii,subspace_idx[nn]]/dE_mm)
                    
        return Heff_0
import numpy as np
#from sympy.physics.wigner import wigner_3j,wigner_6j,wigner_9j
from sympy.physics.wigner import wigner_3j as wigner_3j_sympy
from sympy.physics.wigner import wigner_6j as wigner_6j_sympy
from sympy.physics.wigner import wigner_9j
from sympy.physics.hydrogen import R_nl
from sympy.physics.quantum.cg import CG as CG_sympy

from sympy.utilities import lambdify
#from sympy.functions.special.spherical_harmonics import Ynm
import scipy as sp
import scipy.integrate
import scipy.interpolate
import scipy.constants as cs
import scipy.optimize

import sympy

import time,os

import functools, hashlib

# from .constants import *
# from .single_basis import *

# from .utils import *

from .single_basis import *
from .defects import *

class Hydrogen:
    """
    Base class from which other atoms are derived.
    
    The Hydrogen class has no quantum defects, corrections to the core potential, or spin-orbit coupling.
    Its radial_wavefunction is computed analytically. Its main purpose is to calculate the properties of
    circular Rydberg states.
    
    States are specified using |n,l,m> quantum numbers.
    
    Arguments
    =========
    
    None.
    
    Examples
    ========
    
    >>> H = rydcalc.Hydrogen()
    >>> st1 = H.get_state((50,0,0)) # |nlm> = |50,0,0>
    >>> st2 = H.get_state((50,1,0)) # |50,1,0>
    >>> me = H.get_multipole_me(st1,st2,k=1)
    
    """
    
    name = 'H'
    mass = 1
    
    radial_wavefunction_type = 'analytic'
    
    channels = []
    
    init_dipole_db = False

    citations = []
    
    def __init__(self):
        """
        Initialize a Hydrogen atom instance.
        """
        #self.dbm = db_manager('master_db.db')
        
        self.mass_kg = cs.physical_constants['atomic mass constant'][0] * self.mass
        self.mu = (1-cs.electron_mass/self.mass_kg)#self.mass_kg / (self.mass_kg + cs.electron_mass) #(self.mass_kg - cs.electron_mass) / self.mass_kg
        
        self.core = core_state((0,0,0,0,0),0,tt='sljif',config='p',potential = None)
        self.me_db_keys = ['n','l']
        
        db_keys_all = [s + '_1' for s in self.me_db_keys] + [s + '_2' for s in self.me_db_keys] + ['k']
        #self.me_table = self.dbm.add_table(self.name,db_keys_all,['dme'],'dme_%s.npy' % self.name)
        
        self.hashval = id(self) #hash(self.name + str(self.mass))
        
    def __eq__(self,other):
        
        if self.hashval == other.hashval:
            return True
        else:
            if self.name == other.name and self.mass==other.mass:
                return True
        return False
    
    def __hash__(self):
        return self.hashval
        
    def get_energy_Hz_from_defect(self, n, delta):
        """
        Calculate the energy in Hz, corrected for reduced mass.

        Parameters:
            n (int): Principal quantum number.
            delta (float): Quantum defect.

        Returns:
            float: Energy in Hz.
        """
        R = cs.Rydberg * cs.c  # 3289.82119466e12
        return -2 * R * self.mu / (2 * (n - delta) ** 2)
    
    
    def get_energy_Hz(self, st):
        """
        Calculates the energy in for a given state.

        Parameters:
            st (state): The state object.

        Returns:
            float: The energy in Hz.
        """
        R = cs.Rydberg * cs.c  # 3289.82119466e12

        return 2 * R * self.get_energy_au(st)

    
    def get_energy_au(self, st):
        """
        Calculate the energy in atomic units for a given state of the hydrogen atom.

        Parameters:
            st (state): The state object.

        Returns:
            float: The energy in atomic units.

        Notes:
            The energy is calculated using the formula: -mu / (2 * st.n**2), where mu is the reduced mass
            and st.n is the principal quantum number of the state.
        """
        return -self.mu / (2 * st.n**2)

    
    def radial_wavefunction(self, st, r, force_log=False, ch=None):
        """
        Return the radial wavefunction for a given state, r*R(r).
        
        Parameters:
            - st: The state for which to calculate the radial wavefunction.
            - r: The radial coordinate(s) at which to evaluate the wavefunction. Can be a single value or an array-like object.
            - force_log: If True, use a logarithmic implementation for large values of nn+ll. Default is False.
            - ch: The channel to be used for MQDT states. If None, assumed to be a single-channel state.
            
        Returns:
            The evaluated radial wavefunction(s) at the given radial coordinate(s).
            
        Note:
            Units for input are in Bohr radii, which are scaled for reduced mass inside this function.
            The analytic expressions used here are in units where the length scale is corrected for the reduced mass.
            Since the rest of the program works in atomic units, the input is adjusted accordingly.
        """
        
        if ch is None:
            ch = st.channels[0]
            
        nn = st.n
        ll = ch.l
        
        if type(r) == type([1,2]):
            r = np.array(r)
        
        # The analytic expressions used here are in units where the length scale is corrected for the reduced mass. Since the rest
        # of the program works in atomic units, want to adjust.
        rs = r * self.mu
            
        # somehow the sympy function R_nl does not really evaluate well for n>35, so we just implement it directly
    
    #         r = sympy.symbols('r')
    #         radfn = sympy.lambdify(r,R_nl(n,l,r,Z=1),modules="numpy")
    
        # the grouping matters... you the 1/2n has to be outside of the parens with the factorials to avoid overflow
        # errors somehow... maybe it has to do with numpy knowing how to divide two big integers if they are integers
        # but not floats, which ia makes them when you multiply by 2n?
        
        if (nn+ll < 150) and (not force_log):
            prefactor = np.sqrt((2/(nn/self.mu))**3 * 1/(2*nn) * (scipy.special.factorial(int(nn-ll-1))/(scipy.special.factorial(int(nn+ll)))))
            return prefactor*r*sp.special.eval_genlaguerre(int(nn-ll-1),int(2*ll+1),2*rs/nn)*np.exp(-rs/nn)*(2*rs/nn)**ll
        else:
            # in this case, the prefactors are still too big to work with. The final answer is not, so we can compute the log, sum, and then exp
            logparts = -rs/nn + np.log(r)
            logparts += 0.5*(3*np.log(2/nn) + scipy.special.gammaln((nn-ll-1) + 1) - np.log(2*nn) - scipy.special.gammaln((nn+ll) + 1) )
            logparts += ll*np.log(2*rs/nn)
        
            return sp.special.eval_genlaguerre(int(nn-ll-1),int(2*ll+1),2*rs/nn) * np.exp(logparts)
            
            
            
    def radial_integral(self,s1,s2,k=1,nquad=500,overlap=False,operator=None,s1_ch=None,s2_ch=None):
        """Integrate the radial wavefunction, in units x = sqrt(r).

        Args:
            s1 (State): The first state for the radial integral.
            s2 (State): The second state for the radial integral.
            k (int, optional): The multipole rank. Defaults to 1.
            nquad (int, optional): The number of quadrature points. Defaults to 500.
            overlap (bool, optional): Specifies whether to calculate the overlap integral instead of <r>. 
                                        Defaults to False.
            operator (function, optional): The operator function to be applied in the integral. 
                                            Defaults to None.
            s1_ch (str, optional): The channel for the first state if MQDT state. Defaults to None.
            s2_ch (str, optional): The channel for the second state if MQDT state. Defaults to None.

        Returns:
            float: The value of the radial integral.
        """
        # this code works but we have commented it out for now because it needs to be implemented
        # in alkali atom class as well for tests to pass.
        # db_vals = [getattr(s1,x) for x in self.me_db_keys] + [getattr(s2,x) for x in self.me_db_keys] + [k]
        # db_res = self.me_table.query(db_vals)
        
        # if db_res is not None:
        #     return db_res
        
        nmin = min(s1.n,s2.n)
        
        if s1_ch is None:
            s1_ch = s1.channels[0]
            s1_ch_idx = 0
        else:
            s1_ch_idx = s1.get_channel_idx(s1_ch)
        
        if s2_ch is None:
            s2_ch = s2.channels[0]
            s2_ch_idx = 0
        else:
            s2_ch_idx = s2.get_channel_idx(s2_ch)
        
        if nmin < 120:
            nmin=120 # force fixed quad to evaluate same points every time
        # limit = np.sqrt(2*nmin*(nmin+15) - 1)
        #limit = 2.5*max(self.n,other.n)
        #print(limit)
        
        if self.radial_wavefunction_type == 'numeric':
            # the point of these calls is to force radial_wavefunction to evaluate
            # the wavefunction so we can get the domain
            test = self.radial_wavefunction(s1,s1.n**2,ch=s1_ch)
            test2 = self.radial_wavefunction(s2,s2.n**2,ch=s2_ch)

            limit_in = np.sqrt(max(s1.wf_x_min[s1_ch_idx],s2.wf_x_min[s2_ch_idx]) + 0.01)
            limit_out = np.sqrt(min(s1.wf_x_max[s1_ch_idx],s2.wf_x_max[s2_ch_idx]) - 0.01)
        else:
            limit_in = 0
            limit_out = np.sqrt(2*nmin*(nmin+15) - 1)
        
        
        if operator is None:
            if overlap is True:
                fn = lambda x: 2*x * self.radial_wavefunction(s1,x**2,ch=s1_ch) * self.radial_wavefunction(s2,x**2,ch=s2_ch)
            else:
                fn = lambda x: 2*x * x**(2*k) * self.radial_wavefunction(s1,x**2,ch=s1_ch) * self.radial_wavefunction(s2,x**2,ch=s2_ch)
        else:
            fn = lambda x: 2*x * operator(x**2,k=k) * self.radial_wavefunction(s1,x**2,ch=s1_ch) * self.radial_wavefunction(s2,x**2,ch=s2_ch)
        
        v,e = scipy.integrate.fixed_quad(fn,limit_in,limit_out,n=nquad)
        
        # this code works but we have commented it out for now because it needs to be implemented
        # in alkali atom class as well for tests to pass.
        #self.me_table.insert(db_vals+[v])
        
        return v
    
    def get_quantum_defect_model(self,qns):
        return defect_model(0,lambda qns: True,True,True,True)
    
    def get_quantum_defect(self,qns):
        
        return self.get_quantum_defect_model(qns).get_defect(qns)
        
        print("in get_quantum_defect(): no default set qns=",qns)
        return 0
    
    # this caches instances of states, which allows them to be reused across a single computational session.
    # it significantly speeds up large calculations (because wavefunctions are cached in the states)
    # but creates the potential for non-correctness if you are playing with the quantum defect models -- it is
    # not guaranteed that changes will be passed through to already existing state objects.
    @functools.lru_cache(maxsize=1024)
    def get_state(self,qn,tt='nlm',whittaker_wfct=False):
        """
        Retrieve or create a quantum state based on quantum numbers and type.

        Parameters:
        qn : tuple
            Quantum numbers specifying the state.
        tt : str, optional
            Type of quantum numbers provided, defaults to 'nlm'.

        whittaker_wf : bool, optional
            If True, computes the wavefunction using the generalized Coulomb Whittaker function (see self.whittaker_wfct).
            If False (default), computes the wavefunction numerically using the Numerov method.

        Returns:
        state object or None
            The quantum state object if valid quantum numbers are provided, otherwise None.
        """
        # first, find suitable channel
        if tt == 'nlm' or len(qn)==3:
            
            n = qn[0]
            l = qn[1]
            j = l
            s = 0
            m = qn[2]
            
            # for defect model
            qns = {'n': n, 'l':l ,'j': j}
            
            if l < 0 or l >= n or np.abs(m) > l:
                return None
            
        elif tt == 'nljm' or len(qn)==4:
            
            n = qn[0]
            l = qn[1]
            j = qn[2]
            s = 1/2
            m = qn[3]
            
            # for defect model
            qns = {'n': n, 'l': l, 'j': j}
            
            if s < 0 or l < 0 or l >= n or np.abs(m) > j or j < np.abs(l-s) or j > l+s:
                return None
            
        else:
            print("tt=",tt," not supported by H.get_state")
        
        my_ch = None
        
        # search through defined channels in the atom for a suitable one
        for ch in self.channels:
            
            if ch.l == l and ch.j == j and ch.s == s:
                my_ch = ch
                break
            
        # if we didn't find a channel, make a new one
        if my_ch is None:
            my_ch = channel(self.core,(s,l,j),tt='slj')
            self.channels.append(my_ch)
            
        defect_model = self.get_quantum_defect_model(qns)
        defect = defect_model.get_defect(qns)
        energy_Hz = self.get_energy_Hz_from_defect(n,defect)

        if tt == 'nljm' or len(qn)==4:
            energy_Hz = energy_Hz + self.hydrogenic_correction_Hz(qns,my_ch,defect_model.corrections)
            
        #__init__(self,atom,qn,channel,energy = None,tt='npfm'):
        st = state(self,(n,(-1)**l,j,m),my_ch,energy_Hz = energy_Hz)
        st.nu = n - defect
        st.nub = st.nu
        st.whittaker_wfct = whittaker_wfct
        return st
        
    def repr_state(self,st):
        """ generate a nice ket for printing """
        return "|%s:%d,%d,%d>" % (self.name,st.n,st.channels[0].l,st.m)
        
    def get_g(self, st):
        """Return the lande g factor for the state.

        Args:
            st (State): The state for which to calculate the lande g factor.

        Returns:
            float: The lande g factor for the state.
        """
        return 1  # orbital g factor
    
    def get_magnetic_me(self, st, other):
        """ Helper function for backward compatibility with function in AlkaliAtom. """
        if st == other:
            muB = cs.physical_constants['Bohr magneton'][0]
            return self.get_g(st) * muB
        else:
            return 0

    #@functools.lru_cache(maxsize=4096)
    def allowed_multipole(self,st,other,k=1,qIn=None):
            """Estimate whether a rank-k electric multipole transition is allowed.

            Args:
                st (object): The initial state of the transition.
                other (object): The final state of the transition.
                k (int, optional): The rank of the electric multipole transition. Defaults to 1.
                qIn (array-like, optional): The allowed polarization values. Defaults to None.

            Returns:
                bool: True if the transition is allowed, False otherwise.
            """
            
            if qIn is None:
                qIn = np.arange(-k,k+1)
            
            
            if other.f >= np.abs(st.f - k) and other.f <= k + st.f \
                and other.m - st.m in qIn \
                    and st.parity*other.parity == (-1)**k:
                return True
            
            return False
        
    def diamagnetic_int(self, st, other):
        """
        Calculate the matrix element of the diamagnetic interaction between two states, for a z-oriented magnetic field.
        
        Args:
            st (State): The final state.
            other (State): The initial state.
        
        Returns:
            The matrix element in units of J/T**2.
        
        Notes:
            The diamagnetic interaction vanishes if m != mdash, but it couples channels with \Delta ch.l = 0, +/-2.
        """

        if st.m != other.m:
            return 0

        # W-E theorem
        # NB: the abs is needed to avoid python complaining about (-1)**(-1), but I think it doesn't change anything
        prefactor_WE_0 = (-1) ** np.abs(other.f - other.m) * wigner_3j(other.f, 0, st.f, -other.m, 0, st.m)
        prefactor_WE_2 = (-1) ** np.abs(other.f - other.m) * wigner_3j(other.f, 2, st.f, -other.m, 0, st.m)


        reduced_0 = 0
        reduced_2 = 0

        for ii, (Ai, chi) in enumerate(zip(st.Ai, st.channels)):
            for jj, (Aj, chj) in enumerate(zip(other.Ai, other.channels)):

                # if either of channels is a state that we can't compute wavefunctions for (ie, core excited state), then skip
                if chi.no_me or chj.no_me:
                    continue

                # if inner product of core states is zero, we're done
                if chi.core != chj.core:
                    continue

                if np.abs(chi.l - chj.l) != 0 and np.abs(chi.l - chj.l) != 2:
                    continue

                # first apply Edmonds 7.1.8 to convert reduced ME in F basis to Rydberg electron quantum numbers (s,l,j) belonging to channel
                # <j1' j2' J' || U(k) || j1, j2, J>
                # j1' = chj.core.f
                # j2' = chj.j
                # J' = other.f
                # j1 = chi.core.f
                # j2 = chi.j
                # J = st.f
                prefactor_FtoCh_0 = (-1) ** (chi.core.f + chi.j + other.f + 0) * np.sqrt(
                    (2 * st.f + 1) * (2 * other.f + 1)) * wigner_6j(chj.j, other.f, chi.core.f, st.f, chi.j, 0)

                prefactor_FtoCh_2 = (-1) ** (chi.core.f + chi.j + other.f + 2) * np.sqrt(
                    (2 * st.f + 1) * (2 * other.f + 1)) * wigner_6j(chj.j, other.f, chi.core.f, st.f, chi.j, 2)

                # then apply 7.1.8 again to convert reduced ME from fine-structure basis (on Ryd. electron) to L basis
                # <j1' j2' J' || U(k) || j1, j2, J>
                # j1' = chj.s
                # j2' = chj.l
                # J' = chj.j
                # j1 = chi.s
                # j2 = chi.l
                # J = chi.j
                prefactor_JtoL_0 = (-1)**(chi.s + chi.l + chj.j + 0) * np.sqrt((2*chi.j+1)*(2*chj.j+1)) * wigner_6j(chj.l, chj.j, chi.s, chi.j, chi.l, 0)
                prefactor_JtoL_2 = (-1) ** (chi.s + chi.l + chj.j + 2) * np.sqrt((2 * chi.j + 1) * (2 * chj.j + 1)) * wigner_6j(chj.l, chj.j, chi.s, chi.j, chi.l, 2)

                # use 7.1.7, acting on system 1, this gives consistent sign with our old code
                # <j1' j2' J' || U(k) || j1, j2, J>
                # j1' = chj.l
                # j2' = chj.s
                # J' = chj.j
                # j1 = chi.l
                # j2 = chi.s
                # J = chi.j
                #prefactor_JtoL_0 = (-1) ** (chj.l + chi.s + chi.j + 0) * np.sqrt((2 * chi.j + 1) * (2 * chj.j + 1)) * wigner_6j(chj.l, chj.j, chi.s, chi.j, chi.l, 0)
                #prefactor_JtoL_2 = (-1) ** (chj.l + chi.s + chi.j + 2) * np.sqrt((2 * chi.j + 1) * (2 * chj.j + 1)) * wigner_6j(chj.l, chj.j, chi.s, chi.j, chi.l, 2)

                # prefactor on radial matrix element
                redmat_prefactor_0 = (-1) ** chj.l * np.sqrt((2 * chi.l + 1) * (2 * chj.l + 1)) * wigner_3j(chj.l, 0,chi.l, 0, 0,0)
                redmat_prefactor_2 = (-1) ** chj.l * np.sqrt((2 * chi.l + 1) * (2 * chj.l + 1)) * wigner_3j(chj.l, 2, chi.l, 0, 0, 0)

                redmat = self.radial_integral(st, other, k=2, operator=None, s1_ch=chi, s2_ch=chj)

                reduced_0 += np.conjugate(Aj) * Ai * prefactor_FtoCh_0 * prefactor_JtoL_0 * redmat_prefactor_0 * redmat
                reduced_2 += np.conjugate(Aj) * Ai * prefactor_FtoCh_2 * prefactor_JtoL_2 * redmat_prefactor_2 * redmat

                # print(np.conjugate(Aj) * Ai,prefactor_WE,prefactor_FtoCh,prefactor_JtoL,redmat_prefactor,redmat)* prefactor_WE_0* prefactor_WE_2

        diam = (a0**2) * (cs.e**2) / (12 * cs.electron_mass * self.mu) * (reduced_0 * prefactor_WE_0 - (reduced_2 * prefactor_WE_2))

        return float(diam)
    
    #@functools.lru_cache(maxsize=2048)
    def get_multipole_me(self, st, other, k=1, qIn=None, operator=None):
        """ 
        Calculate the multipole matrix element between two states. Note that this only considers the matrix element from the _outer_
        electron.
        
        Parameters:
        - st: Initial state.
        - other: Final state.
        - k: Multipole order (default is 1).
        - qIn: Optional argument restricting polarization (default is None).
        - operator: Operator to be used in the radial integral, ie, a ponderomotive object (default is None).
        
        Returns:
        The multipole matrix element between the initial and final states.
        The answer is in atomic units, e*a0**k.
        """

        if qIn is None:
            qIn = np.arange(-k, k + 1)

        q = other.m - st.m

        if not (q in qIn):
            return 0.0

        # W-E theorem
        # NB: the abs is needed to avoid python complaining about (-1)**(-1), but I think it doesn't change anything
        prefactor_WE = (-1) ** np.abs(other.f - other.m) * wigner_3j(other.f, k, st.f, -other.m, q, st.m)


        reduced_me = 0

        for ii, (Ai, chi) in enumerate(zip(st.Ai, st.channels)):
            for jj, (Aj, chj) in enumerate(zip(other.Ai, other.channels)):

                # if either of channels is a state that we can't compute wavefunctions for (ie, core excited state), then skip
                if chi.no_me or chj.no_me:
                    continue

                # if inner product of core states is zero, we're done
                if chi.core != chj.core:
                    continue

                # first apply Edmonds 7.1.8 to convert reduced ME in F basis to Rydberg electron quantum numbers (s,l,j) belonging to channel
                # <j1' j2' J' || U(k) || j1, j2, J>
                # j1' = chj.core.f
                # j2' = chj.j
                # J' = other.f
                # j1 = chi.core.f
                # j2 = chi.j
                # J = st.f
                prefactor_FtoCh = (-1) ** (chi.core.f + chi.j + other.f + k) * np.sqrt(
                    (2 * st.f + 1) * (2 * other.f + 1)) * wigner_6j(chj.j, other.f, chi.core.f, st.f, chi.j, k)

                # then apply 7.1.8 again to convert reduced ME from fine-structure basis (on Ryd. electron) to L basis
                # <j1' j2' J' || U(k) || j1, j2, J>
                # j1' = chj.s
                # j2' = chj.l
                # J' = chj.j
                # j1 = chi.s
                # j2 = chi.l
                # J = chi.j
                prefactor_JtoL = (-1)**(chi.s + chi.l + chj.j + k) * np.sqrt((2*chi.j+1)*(2*chj.j+1)) * wigner_6j(chj.l, chj.j, chi.s, chi.j, chi.l, k)

                # use 7.1.7, acting on system 1, this gives consistent sign with our old code
                # <j1' j2' J' || U(k) || j1, j2, J>
                # j1' = chj.l
                # j2' = chj.s
                # J' = chj.j
                # j1 = chi.l
                # j2 = chi.s
                # J = chi.j
                #prefactor_JtoL = (-1) ** (chj.l + chi.s + chi.j + k) * np.sqrt(
                #    (2 * chi.j + 1) * (2 * chj.j + 1)) * wigner_6j(chj.l, chj.j, chi.s, chi.j, chi.l, k)

                # prefactor on radial matrix element
                redmat_prefactor = (-1) ** chj.l * np.sqrt((2 * chi.l + 1) * (2 * chj.l + 1)) * wigner_3j(chj.l, k,chi.l, 0, 0,0)


                redmat = self.radial_integral(st, other, k=k, operator=operator, s1_ch=chi, s2_ch=chj)

                reduced_me += np.conjugate(Aj) * Ai * prefactor_FtoCh * prefactor_JtoL * redmat_prefactor * redmat

                # print(np.conjugate(Aj) * Ai,prefactor_WE,prefactor_FtoCh,prefactor_JtoL,redmat_prefactor,redmat)

        me = reduced_me * prefactor_WE

        return float(me)
    
    def get_nearby(self,st,include_opts={}):
        """Generate a list of quantum number tuples specifying nearby states for sb.fill().
        For Hydrogen, these are (n,l,m)

        Args:
            st (State): The reference state.
            include_opts (dict): Optional dictionary to override options for including states.
        
        Returns:
            list: A list of tuples specifying nearby quantum number states.
        
        Notes:
            - Does not create states or check valid qn, just returns list of tuples.
            - include_opts can override options in terms of what states are included.
            - It's a little messy to decide which options should be handled here vs. in single_basis.
                Decision for now is to have all quantum numbers here but selection rules/energy cuts
                in single_basis to avoid duplication of code.
        """
        
        ret = []
        
        o = {'dn': 2, 'dl': 2, 'dm': 1}
        
        for k,v in include_opts.items():
            o[k] = v
            
        for n in np.arange(st.n-o['dn'],st.n+o['dn']+1):
            for l in np.arange(st.f-o['dl'],st.f+o['dl']+1):
                for m in np.arange(st.m-o['dm'],st.m+o['dm']+1):
                    
                    #ret.append((n,l,m))
                    new_st = self.get_state((n,l,m))
                    if new_st is not None:
                        ret.append(new_st)
        
        return ret
    
    def partial_decay(self,st,other,env,return_nth=False):
        """Return the partial decay rate from self->other.

        Args:
            st (State): The initial state (n, l, m) of the transition.
            other (State): The final state (n, l, m) of the transition.
            env (Environment): The environment in which the transition occurs.
            return_nth (bool, optional): Whether to return the thermal occupation factor. Defaults to False.

        Returns:
            float: The partial decay rate.

        Raises:
            None

        Notes:
            The partial decay rate is calculated using the formula:
            gamma = omega^3 * d^2 / (3*pi*eps0*hbar*c^3)

        """
        
        omega = (2*np.pi)*(st.get_energy_Hz()-other.get_energy_Hz())
        
        if env.T_K == 0:
            n_th = 0
        else:
            if omega != 0:
                A = (cs.Boltzmann*env.T_K)/(cs.hbar*np.abs(omega))
                n_th = 1/(np.exp(1/A) - 1)
            else:
                n_th = 1e6 # this probably shouldn't happen...?
            

        # get matrix element
        total_me = st.get_multipole_me(other,k=1) #getdipole(n1,l1,m1,n2,l2,m2)
        q = other.m - st.m
        
        # get thermal occupation factor
        th_factor = n_th if omega < 0 else (1+n_th)
        
        # calculate decay rate
        unit_factor = (cs.elementary_charge * a0)**2 / (3*np.pi*cs.epsilon_0*cs.hbar*cs.c**3)

        gamma = env.ldos(np.abs(omega)/(2*np.pi),q)*th_factor * np.abs(omega)**3 * total_me**2 * unit_factor
        
        if return_nth:
            return gamma,n_th
        
        return gamma
    
    def total_decay(self, st, env, printstates=False, include_fn=lambda s: True):
        """
        Calculate the total decay rate from the current state to other states.

        Parameters:
        - st: The current state.
        - env: The environment.
        - printstates: Whether to print the states we decay to (default: False).
        - include_fn: A function that determines which states to include in the decay calculation, passed as an argument to single_basis.fill (default: lambda s: True).

        Returns:
        - If printstates is True, returns a list of final states and their decay rates.
        - If printstates is False, returns the total decay rate.
        """
        gamma_tot = 0
        
        if printstates:
            finalstates = []
            
        # construct basis of states we might decay to
          
        sb = single_basis()
        
        nmax = st.n + 10
        
        # by setting dnu = n, we will include all allowed states starting from n=0
        sb.fill(st,{'dn': st.n, 'dl': 1, 'dm': 1, 'include_fn': lambda s,s0: True if s.n < nmax else False})
            
        # compute lifetime
                            
        for s in sb.states:
            pd = self.partial_decay(st,s,env)

            if pd != 0 and printstates:
                finalstates.append([s,pd])
                #print("(%d,%d,%d): %f"%(n2,l2,m2,pd))
                
            gamma_tot += pd
        
        if printstates:
            return finalstates
            # print(finalstates)
            # finalstates = np.array(finalstates)
            # finalstates = finalstates[np.flip(np.argsort(finalstates[:,3]))]
            # return finalstates
        else:
            return gamma_tot
    
    def potential_ponderomotive(self,lam_nm):
        """
        Calculate the ponderomotive potential shift for a spatially uniform laser field of a given wavelength.

        Parameters:
        lam_nm : float
            Wavelength in nanometers.

        Returns:
        float
            The ponderomotive shift in Hz/(W/m^2).
        """
        omega = 2*np.pi*cs.c/(lam_nm*1e-9)
        
        return -cs.e**2/(2*cs.electron_mass * omega**2 * cs.epsilon_0 * cs.c)/cs.h


    def hydrogenic_correction_Hz(self,qns,ch,corrections):
        """
        Calculate hydrogenic corrections in Hz for given quantum numbers and core properties.

        Parameters:
        qns : dict
            Quantum numbers of the state.
        ch : core object
            Core properties affecting the correction.

        Returns:
        float
            Total correction in Hz.
        """
        corr = 0

        if corrections["polcorrection"] == True:
            corr += self.energy_static_dip_Hz(qns,ch.core)
            corr += self.energy_static_quad_Hz(qns,ch.core)

        if corrections["SOcorrection"] == True:
            corr += self.hydrogenic_spin_orbit_Hz(qns)

        if corrections["relcorrection"] == True:
            corr += self.hydrogenic_relativistic_Hz(qns)

        return corr

    def energy_static_dip_Hz(self,qns,core):

        n = qns['n']
        l = qns['l']

        return -(1/2)*self.mu**4*core.alpha_d_a03*((3*n**2-l*(l+1))/(2*n**5*(l-1/2)*l*(l+1/2)*(l+1)*(l+3/2)))*cs.physical_constants['hartree-hertz relationship'][0]

    def energy_static_quad_Hz(self,qns,core):

        n = qns['n']
        l = qns['l']

        return -(1/2)*self.mu**6*core.alpha_q_a05*((35*n**4-n**2*(30*l*(l+1)-25)+3*(l-1)*l*(l+1)*(l+2))/(8*n**7*(l-3/2)*(l-1)*(l-1/2)*l*(l+1/2)*(l+1)*(l+3/2)*(l+2)*(l+5/2)))*cs.physical_constants['hartree-hertz relationship'][0]

    def hydrogenic_spin_orbit_Hz(self,qns):

        n = qns['n']
        l = qns['l']
        j = qns['j']

        splitting = self.mu*(-cs.physical_constants['electron g factor'][0]/2)*((cs.physical_constants['fine-structure constant'][0]**2))/(2*l*(l+1)*n**3)*cs.physical_constants['hartree-hertz relationship'][0]

        if j == l + 1/2:
            return (l / (2*l+1))*splitting
        elif j == l - 1/2:
            return -((l+1)/(2*l+1))*splitting
        else:
            return 0

    def hydrogenic_relativistic_Hz(self,qns):

        n = qns['n']
        l = qns['l']

        return -(self.mu*(cs.physical_constants['fine-structure constant'][0]**2)/(8*n**4))*(-3+4*n/(l+1/2))*cs.physical_constants['hartree-hertz relationship'][0]



                
 
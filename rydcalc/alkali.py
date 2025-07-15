import numpy as np
import scipy as sp
import scipy.integrate
import scipy.interpolate
import scipy.constants as cs
import scipy.optimize
from mpmath import whitw

import sympy

import sqlite3

import time,os

import functools, hashlib

from .constants import *
from .hydrogen import Hydrogen
from .single_basis import *
from .defects import *



class AlkaliAtom(Hydrogen):
    """
    Class to represent an alkali atom. Inherits from Hydrogen class.

    There are several key differences to the Hydrogen class:
        - State energies are calculated from quantum defects
        - Wavefunctions are computed numerically, including optional model potential for ion core.
        - Matrix elements are calculated numerically
        - Quantum numbers are used including spin-orbit coupling, ie, nljm.

    """
    
    name = ''
    mass = 0
    
    dipole_db_file = None
    dipole_data_file = None
    dataFolder = rydcalc_data_folder
    
    radial_wavefunction_type = 'numeric'
    
    Z = 0.0       #: Atomic number
    I = 0.0       #: Nuclear spin
    
    channels = []

    transitions_neutral = []
    
    ground_state_n = 0
    
    def __init__(self,cpp_numerov=False,use_db=True):
        """
        Initialize an instance of the AlkaliAtom class.

        Parameters:
        - cpp_numerov (bool): Flag indicating whether to use the C++ implementation of NumerovWavefunction.
        - use_db (bool): Flag indicating whether to use the dipole_db_file to save matrix elements between sessions.

        Returns:
        None
        """
        self.cpp_numerov = cpp_numerov
        
        if self.cpp_numerov:
            from .arc_c_extensions import NumerovWavefunction
            self.NumerovWavefunction = NumerovWavefunction
        
        self.mass_kg = cs.physical_constants['atomic mass constant'][0] * self.mass
        
        self.hashval = hash(self.name + str(self.mass))
        
        self.mu = (1 - cs.electron_mass / self.mass_kg)
        
        if self.dipole_db_file is not None and use_db:
            self._init_db()
    
    def _init_db(self):

        self.conn = sqlite3.connect(os.path.join(self.dataFolder,self.dipole_db_file))
        self.c = self.conn.cursor()
        
        # save to SQLite database
        try:
            
            # see if we have a dipoleME table
            self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='dipoleME';''')

            if (self.c.fetchone()[0] == 0):
            
                query_str = self._dipole_db_query(None,None,'i')
                self.c.execute("CREATE TABLE IF NOT EXISTS dipoleME %s" % query_str)
                self.conn.commit()
                
                self._db_load_from_file()
                
            else:
                dbsize = self._db_size()
                print("Reloaded db of size %d" % dbsize)
                
            self.init_dipole_db = True
            
        except sqlite3.Error as e:
            print("Error while loading precalculated values into the database")
            print(e)
            exit()
    
    def _db_size(self):
        self.c.execute('''Select count(*) from dipoleME''')
        dbsize = self.c.fetchone()[0]
        
        return dbsize
    
    def _db_load_from_file(self):
        
        datafile = os.path.join(self.dataFolder,self.dipole_data_file)
        
        try:
            self.data = np.load(datafile,encoding='latin1', allow_pickle=True)
        except:
            print("Error loading data file:", datafile)
            return
            #exit()
                
        # create table
        print("Loaded data file with %d values. Creating table..." % len(self.data))

        if (len(self.data) > 0):
            query_str = self._dipole_db_query(None,None,'wf')
            self.c.executemany('INSERT INTO dipoleME VALUES %s' % query_str, self.data)
            
        self.conn.commit()
        
        print("Populated db with %d values", len(self.data))
        
        self.c.execute('''Select count(*) from dipoleME''')
        dbsize = self.c.fetchone()[0]
        
        print("Confirmed db with %d values", len(self.data))
    
    def _db_save_to_file(self,file=None):
        
        if file is None:
            # save to default location
            file = self.dipole_data_file
        
        datafile = os.path.join(self.dataFolder,file)
        
        query_str = self._dipole_db_query(None,None,'rf')
        
        self.c.execute("SELECT %s,dme from dipoleME" % query_str)
        
        self.data = self.c.fetchall()
        
        np.save(datafile,self.data,allow_pickle=True)
        
        print("Saved %d values to file %s", len(self.data), datafile)
        
        
    def potential_numerov(self,st,ch,x):
        """ the core potential for the python numerov solver (see J.D. Prichard PhD thesis p. 17) 
        [borrowed from ARC] """
        r = x**2
        
        #threshold_au = getattr(st,'threshold_au',0)
        threshold_au = ch.core.Ei_au

        #return -3. / (4. * r) + 4*r*(2*self.mu*(st.energy_au - (-1/r)) - st.l * (st.l + 1) / (r**2))
        return -3. / (4. * r) + 4*r*(2*self.mu*(self.get_energy_au(st) - threshold_au - ch.core.potential.potential(ch,r)) - ch.l * (ch.l + 1) / (r**2))
  
    
    def radial_wavefunction(self,st,r,ch = None, whittaker_wf = False):
        """
        Compute the radial wavefunction for a given state and radius. The results are cached within the
        st object, so do not need to be computed again in the same session.

        Parameters:
        st : State object
            The state for which the wavefunction is to be computed.
        r : array-like
            The radial distance(s) at which the wavefunction is evaluated.
        ch : Channel object, optional
            The MQDT channel for which the wavefunction is computed. If None, the first channel of the state is used.

        whittaker_wf : bool, optional
            If True, computes the wavefunction using the generalized Coulomb Whittaker function (see self.whittaker_wfct).
            If False (default), computes the wavefunction numerically using the Numerov method.

        Returns:
        float or ndarray:
            The value of the wavefunction at the given radii.
        """
        
        whittaker_wf = st.whittaker_wfct
        
        if ch is None:
            ch = st.channels[0]
            
        ch_idx = st.get_channel_idx(ch)
        
        if getattr(st,'wf_interp',None) is None:
            
            st.wf_interp = [None]*len(st.channels)
            st.wf_x_min = [None]*len(st.channels)
            st.wf_x_max = [None]*len(st.channels)
            
        if st.wf_interp[ch_idx] is None:
            
            st.wf_x,st.wf_y = self.numerov_py(st, ch = ch)
            
            st.wf_x_min[ch_idx] = min(st.wf_x)
            st.wf_x_max[ch_idx] = max(st.wf_x)
            
            if whittaker_wf==False:
                st.wf_interp[ch_idx] = scipy.interpolate.interp1d(st.wf_x,st.wf_y)
            else:
                st.wf_interp[ch_idx] = lambda r : self.whittaker_wfct(st,ch,r)
        
        return st.wf_interp[ch_idx](r)
    
    def _dipole_db_query(self,s1,s2,rwi,me=0):
        
        # Provide strings to query to/from dipole matrix element database.
        # Ideally, this is agnostic about which type of matrix element is being stored,
        # and is the only thing that needs to be modified for different types of atoms
        # with different quantum numbers.
        
        # Given the need to have different strings for creating tables and loading
        # from files, this function has become a bit ugly
        
        # length of this should be 2*(number of quantum numbers) + 1 for matrix element
        insert_str = "(?,?,?,?,?,?,?)"
        
        if rwi=='i':
            # query for initializing database
            query_str = '''(n1 TINYINT UNSIGNED, l1 TINYINT UNSIGNED,
                 j1_x2 TINYINT UNSIGNED,
                 n2 TINYINT UNSIGNED, l2 TINYINT UNSIGNED,
                 j2_x2 TINYINT UNSIGNED,
                 dme DOUBLE,
                 PRIMARY KEY (n1,l1,j1_x2,n2,l2,j2_x2 )
                 )'''

            return query_str
        
        if rwi=='wf':
            # query for writing to db from file
            query_str = insert_str
            
            return query_str
        
        if rwi=='rf':
            # query for read from db to save to file
            query_str = 'n1,l1,j1_x2,n2,l2,j2_x2'
            return query_str
        
        # database is ordered by energy
        if self.get_energy_au(s1) < self.get_energy_au(s2):
            s1_o = s1
            s2_o = s2
        else:
            s1_o = s2
            s2_o = s1
            
        if rwi=='r':
            # query for reading from database
            query_str = "n1= ? AND l1 = ? AND j1_x2 = ? AND n2 = ? AND l2 = ? AND j2_x2 = ?"
            query_dat = (s1_o.n, s1_o.l, int(2*s1_o.j), s2_o.n, s2_o.l, int(2*s2_o.j))
        
        if rwi=='w':
            # query for writing to database (storing matrix element me)
            query_str = insert_str
            query_dat = (s1_o.n, s1_o.l, int(2*s1_o.j), s2_o.n, s2_o.l, int(2*s2_o.j),me)
        
        return query_str, query_dat
    
    def radial_integral(self,s1,s2,k=1,nquad=500,overlap=False,use_db=True,operator=None, s1_ch = None, s2_ch = None):
        """ Calculate the radial integral for the given states s1 and s2 in units in units x = sqrt(r).

        Parameters:
        s1 (State): The first state involved in the radial integral calculation.
        s2 (State): The second state involved in the radial integral calculation.
        k (int, optional): The multipole order of the integral. Defaults to 1.
        nquad (int, optional): The number of quadrature points for numerical integration. Defaults to 500.
        overlap (bool, optional): If True, calculates the overlap integral instead of the radial integral. Defaults to False.
        use_db (bool, optional): If True, attempts to use precomputed values from a database. Defaults to True.
        operator (callable, optional): The operator function to use in the integral calculatio (ie, ponderomotive potential). Defaults to None.
        s1_ch (core, optional): Core properties for state s1. Defaults to None.
        s2_ch (core, optional): Core properties for state s2. Defaults to None.
        
        Returns:
        float: The computed radial integral or overlap integral value.
        """

        if use_db and self.init_dipole_db and k==1 and operator is None:
            # was this calculated before? If it was, retrieve from memory
            # it is somewhat misleading that this table is refered to as dipole matrix elements
            # since it is really radial integrals that do not have any angular terms in them...
            # self.c.execute('''SELECT dme FROM dipoleME WHERE
            #  n1= ? AND l1 = ? AND j1_x2 = ? AND
            #  n2 = ? AND l2 = ? AND j2_x2 = ?''', (s1_o.n, s1_o.l, int(2*s1_o.j), s2_o.n, s2_o.l, int(2*s2_o.j)))
            # dme = self.c.fetchone()
            
            query_str, query_dat = self._dipole_db_query(s1,s2,'r')
            self.c.execute("SELECT dme FROM dipoleME WHERE %s" % query_str, query_dat)
            dme = self.c.fetchone()
            
            #print(s1,s2,dme)
            if (dme):
                return dme[0]
            
        #print("Missed db on ",s1,s2)
            # else:
            #     print('''SELECT dme FROM dipoleME WHERE
            #  n1= ? AND l1 = ? AND j1_x2 = ? AND
            #  n2 = ? AND l2 = ? AND j2_x2 = ?''', (s1_o.n, s1_o.l, int(2*s1_o.j), s2_o.n, s2_o.l, int(2*s2_o.j)))
        
        # if not, call Hydrogen() function to compute numerically
        # note that calls to radial_wavefunction from super().radial_integral
        # still go to the radial_wavefunction() belonging to _this_ class,
        # so this will use the numerov function.
        #return super().radial_integral(s1,s2,nquad=nquad,overlap=overlap)
            
        # Note: this formulation is a little more explicit and is what we need
        # to use if we override AlkaliAtom.radial_integral in a subclass
        # of AlkaliAtom, to skip all the way up the object heirarchy to Hydrogen
        computed = Hydrogen.radial_integral(self,s1,s2,k=k,nquad=nquad,overlap=overlap,operator=operator,s1_ch = s1_ch, s2_ch = s2_ch)
        
        # if we didn't find it in the database, put it in there.
        if use_db and self.init_dipole_db and k==1 and operator is None:
            try:
                # we don't need the whole calculation to abort if this goes wrong, although it shouldn't
                # so we do want to know about it
                query_str, query_dat = self._dipole_db_query(s1,s2,'w',me=computed)
                self.c.execute("INSERT into dipoleME VALUES %s" % query_str, query_dat)
                self.conn.commit()
            except:
                print("Error putting radial integral in db for ", s1, s2, "dme=",dme)
        
        return computed
    
    def get_energy_au(self,st):
        """ this returns energy in atomic units, which has to be corrected for reduced mass """
        #return -self.mu/(2*(st.n-self.get_quantum_defect(st))**2)
        return st.get_energy_au()
    
    def get_quantum_defect_model(self,qns):
        
        for dd in self.defects:
            if dd.is_valid(qns):
                return dd
        
        print("in get_quantum_defect(): no default set qns=",qns)
        return 0
    
    #@functools.lru_cache(maxsize=256)
    def numerov_cpp_wrap(self,*args):
        return self.NumerovWavefunction(*args)
    
    def whittaker_wfct(self,st,ch,r):
        """ Generalized Coulomb wavefunction using Whittaker function """
        nu = 1 / np.sqrt((ch.core.Ei_Hz - st.energy_Hz) / (self.RydConstHz))
        l = ch.l
        def whitw_func(r):
            return whitw(nu,l+1/2,(2*r/(nu)))
        whitw_np = np.vectorize(whitw_func)

        return 1/np.sqrt(nu**2*sp.special.gamma(nu+l+1)*sp.special.gamma(nu-l))*whitw_np(np.array(r))

    
    def numerov_py(self,st,ch):
        """
            This code is borrowed from the Alkali Rydberg Calculator, with minor modifications
            to handle the object structure of rydcalc. Please see the LICENSE file for copyright and license attribution.
            
            Full Python implementation of Numerov integration
            Calculates solution function :math:`rad(r)` with descrete step in
            :math:`r` size of `step`, integrating from `outerLimit` towards the
            `innerLimit` (from outside, inwards) equation
            :math:`\\frac{\\mathrm{d}^2 rad(r)}{\\mathrm{d} r^2} = \
                kfun(r)\\cdot rad(r)`.
            Args:
                innerLimit (float): inner limit of integration
                outerLimit (flaot): outer limit of integration
                kfun (function(double)): pointer to function used in equation (see
                    longer explanation above)
                step: descrete step size for integration
                init1 (float): initial value, `rad`(`outerLimit`+`step`)
                init2 (float): initial value,
                    `rad`(`outerLimit`+:math:`2\\cdot` `step`)
            Returns:
                numpy array of float , numpy array of float, int : :math:`r` (a.u),
                :math:`rad(r)`;
            Note:
                Returned function is not normalized!
            Note:
                If :obj:`AlkaliAtom.cpp_numerov` swich is set to True (default),
                much faster C implementation of the algorithm will be used instead.
                That is recommended option. See documentation installation
                instructions for more details.
        """
        
        init1 = 0.01
        init2 = 0.01
        
        pot = ch.core.potential
        threshold_au = ch.core.Ei_au
        
        if pot.alphaC > 0:
            innerLimit = pot.alphaC**(1/3)
        else:
            innerLimit = 0.05 #self.innerLimit
            
        outerLimit = 2*st.n*(st.n+15)
        step = 0.001
        kfun = lambda x: self.potential_numerov(st,ch,x)
        
        if self.cpp_numerov:
            
            if ch.l < 4 and pot.use_model:
                d = self.numerov_cpp_wrap(
                        innerLimit, outerLimit,
                        step, init1, init2,
                        ch.l, ch.s, ch.j, self.get_energy_au(st) - threshold_au, pot.alphaC, cs.fine_structure,
                        pot.Z,
                        pot.a1[ch.l], pot.a2[ch.l], pot.a3[ch.l], pot.a4[ch.l],
                        pot.rc[ch.l],
                        self.mu)
            else:
                d = self.numerov_cpp_wrap(
                        innerLimit, outerLimit,
                        step, init1, init2,
                        ch.l, ch.s, ch.j, self.get_energy_au(st) - threshold_au, 0, cs.fine_structure,
                        1,
                        0,0,0,0,
                        0,
                        self.mu)
            
            rad = d[1]
            sol = d[0]
            
        else:
            br = int((np.sqrt(outerLimit) - np.sqrt(innerLimit)) / step)
            # integrated wavefunction R(r)*r^{3/4}
            sol = np.zeros(br, dtype=np.dtype('d'))
            # radial coordinate for integration \sqrt(r)
            rad = np.zeros(br, dtype=np.dtype('d'))
        
            br = br - 1
            x = np.sqrt(innerLimit) + step * (br - 1)
            sol[br] = (2. * (1. - 5.0 / 12.0 * step**2 * kfun(x)) * init1
                       - (1. + 1. / 12.0 * step**2 * kfun(x + step)) * init2) /\
                (1 + 1 / 12.0 * step**2 * kfun(x - step))
            rad[br] = x
        
            x = x - step
            br = br - 1
        
            sol[br] = (2. * (1. - 5.0 / 12.0 * step**2 * kfun(x)) * sol[br + 1]
                       - (1. + 1. / 12.0 * step**2 * kfun(x + step)) * init1) /\
                (1 + 1 / 12.0 * step**2 * kfun(x - step))
            rad[br] = x
        
            # check if the function starts diverging  before the innerLimit
            # -> in that case break integration earlier
        
            maxValue = 0.
        
            checkPoint = 0
            fromLastMax = 0
        
            while br > checkPoint:
                br = br - 1
                x = x - step
                sol[br] = (2. * (1. - 5.0 / 12.0 * step**2 * kfun(x)) * sol[br + 1]
                           - (1. + 1. / 12.0 * step**2 * kfun(x + step)) * sol[br + 2]
                           ) /\
                    (1. + 1. / 12.0 * step**2 * kfun(x - step))
                rad[br] = x
                if abs(sol[br] * np.sqrt(x)) > maxValue:
                    maxValue = abs(sol[br] * np.sqrt(x))
                else:
                    fromLastMax += 1
                    if fromLastMax > 50:
                        checkPoint = br
            # now proceed with caution - checking if the divergence starts
            # - if it does, cut earlier
        
            divergencePoint = 0
        
            while (br > 0)and(divergencePoint == 0):
                br = br - 1
                x = x - step
                sol[br] = (2. * (1. - 5.0 / 12.0 * step**2 * kfun(x)) * sol[br + 1]
                           - (1. + 1. / 12.0 * step**2 * kfun(x + step)) * sol[br + 2]
                           ) /\
                    (1. + 1. / 12.0 * step**2 * kfun(x - step))
                rad[br] = x
                if (divergencePoint == 0)and(abs(sol[br] * np.sqrt(x)) > maxValue):
                    divergencePoint = br
                    while (abs(sol[divergencePoint]) > abs(sol[divergencePoint + 1])) \
                            and (divergencePoint < checkPoint):
                        divergencePoint += 1
                    if divergencePoint > checkPoint:
                        print("Numerov error")
                        exit()
        
            br = divergencePoint
            while (br > 0):
                rad[br] = rad[br + 1] - step
                sol[br] = 0
                br -= 1
        
            # convert R(r)*r^{3/4} to  R(r)*r
            sol = np.multiply(sol, np.sqrt(rad))
            # convert \sqrt(r) to r
            rad = np.multiply(rad, rad)
        
        # normalize the wavefunction
        suma = np.trapz(sol**2, x=rad)
        sol = sol / (np.sqrt(suma))
    
        return rad, sol

    @functools.lru_cache(maxsize=2**15)
    def get_state(self,qn,tt='nljm'):

        if qn[0] >= self.ground_state_n:
            return super().get_state(qn,tt='nljm')

    def repr_state(self,st):
        """ generate a nice ket for printing """
        #if st.tt == 'nljm': 
        if st.channels[0].l <= 5:
            return "|%s:%d,%s,%.1f,%.1f>" % (self.name,st.n,['S','P','D','F','G','H'][st.channels[0].l],st.f,st.m)
        else:
            return "|%s:%d,%d,%.1f,%.1f>" % (self.name,st.n,st.channels[0].l,st.f,st.m)

    def get_nearby(self, st, include_opts={}):
        """
        Generate a list of nearby states for constructing basis sets. It takes a reference state and returns
        states that are nearby in energy and satisfy certain quantum number constraints.

        Args:
            st (State): The reference state.
            include_opts (dict, optional): Dictionary to override default options for including states.
                Keys can include 'dn', 'dl', 'dm', to specify ranges for effective principal quantum number,
                orbital angular momentum, and magnetic quantum number, respectively.
                Defaults: {'dn': 2, 'dl': 2, 'dm': 1}.

        Returns:
            list: A list of State objects that are nearby the given state `st` according to the specified
            quantum number ranges and the quantum defect model.

        Notes:
            - This version of the function works for states with with only a single channel in the MQDT formalism.
        """

        qn_list = []

        o = {'dn': 2, 'dl': 2, 'dm': 1}

        for k, v in include_opts.items():
            o[k] = v

        # scan over an n range much bigger than 'dn', to be sure we get states in 'dnu' regardless of quantum defect
        for n in np.arange(st.n - o['dn'] - 5, st.n + o['dn'] + 1 + 5):
            for l in np.arange(st.channels[0].l - o['dl'], st.channels[0].l + o['dl'] + 1):
                for j in np.arange(st.f - o['dl'], st.f + o['dl'] + 1):
                    for m in np.arange(-j,j+1):
                        if int(np.abs(st.m-m))<=o['dm']:
                            qn_list.append((n, l, j, m))

        ret = []

        for qn in qn_list:
            st_new = self.get_state(qn)
            if st_new is not None and np.abs(st_new.nu - st.nu)<o['dn']:
                ret.append(st_new)

        return ret

    def get_magnetic_me(self,st,other):
        """
        Calculate the magnetic dipole matrix element between two states.

        This function computes the magnetic dipole matrix element between two MQDT states `st` and `other`, using the method of
        Robicheaux et al, PRA 97 022508 (2018).

        Args:
            st (State): The initial state of the transition.
            other (State): The final state of the transition.

        Returns:
            float: The magnetic dipole matrix element between the two states in units of the Bohr magneton.

        """

        def lam(x):
            return np.sqrt((2 * x + 1) * (x + 1) * x)

        Ft = st.f
        Ftdash = other.f

        muB = cs.physical_constants['Bohr magneton'][0]
        #muN = cs.physical_constants['nuclear magneton'][0]
        gs = -cs.physical_constants['electron g factor'][0]
        #muI = self.gI * muN

        if st.m != other.m or np.abs(st.f - other.f) > 1 or st.parity != other.parity:
            return 0

        prefactor = (-1) ** (st.f - st.m) * wigner_3j(st.f, 1, other.f, -st.m, 0, other.m)



        if self.core.l < 0 or st.channels[0].l != other.channels[0].l or st.channels[0].s != other.channels[0].s:
            # chi.core.l>=0 to exclude unknown effective core states. implemented with l=-1
            return 0

        ll = self.G1(st.channels[0], other.channels[0], Ft, Ftdash) * self.G2(st.channels[0], other.channels[0]) * lam(st.channels[0].l)
        ss = self.G1(st.channels[0], other.channels[0], Ft, Ftdash) * self.G3(st.channels[0], other.channels[0]) * lam(st.channels[0].s)
        II = self.G4(st.channels[0], other.channels[0], Ft, Ftdash) * self.G5(st.channels[0], other.channels[0]) * lam(self.core.i)
        LL = self.G4(st.channels[0], other.channels[0], Ft, Ftdash) * self.G6(st.channels[0], other.channels[0]) * self.G7(st.channels[0], other.channels[0]) * lam(self.core.l)
        SS = self.G4(st.channels[0], other.channels[0], Ft, Ftdash) * self.G6(st.channels[0], other.channels[0]) * self.G8(st.channels[0], other.channels[0]) * lam(self.core.s)

        stnu = st.nu
        othernu = other.nu

        if stnu == othernu:
            overlap = 1
        else:
            overlap = (2 * np.sqrt(stnu * othernu) / (stnu + othernu)) * (
                        np.sin(np.pi * (stnu - st.channels[0].l) - np.pi * (othernu - other.channels[0].l)) / (
                            np.pi * (stnu - st.channels[0].l) - np.pi * (othernu - other.channels[0].l)))

        reduced_me = overlap * (muB * (LL + ll + gs * (SS + ss)))# - muI * II)

        # need to implement other q for coupling
        # mu+=overlap*Ai*np.conjugate(Aj)*np.conjugate(((-1)**(st.f-st.m))*wigner_3j(st.f,1,other.f,-st.m,0,other.m))*(muB*(LL+ll+gs*(SS+ss))-muI*II)

        me = prefactor * reduced_me
        return me / muB

    def get_g(self, st):
        """ Return the effective g-factor for the state. The magnetic moment is:

            g * uB * m
        """
        if st.m == 0:
            return 0
        else:
            return self.get_magnetic_me(st, st) / (st.m)

    # G1-G8 are helper functions for get_magnetic_me, defined following Robicheaux et al, PRA 97 022508 (2018).
    def G1(self, chi, chj, Ft, Ftdash):

        Jc = chi.core.j
        Jcdash = chj.core.j
        Fc = chi.core.f
        Fcdash = chj.core.f
        j = chi.j
        jdash = chj.j

        if Jc == Jcdash and Fc == Fcdash:
            return (-1) ** (Fc + jdash + Ft + 1) * np.sqrt((2 * Ft + 1) * (2 * Ftdash + 1)) * wigner_6j(j, Ft, Fc, Ftdash, jdash, 1)
        else:
            return 0

    def G2(self, chi, chj):
        j = chi.j
        jdash = chj.j
        l = chi.l
        s = chi.s

        return (-1) ** (s + l + j + 1) * np.sqrt((2 * j + 1) * (2 * jdash + 1)) * wigner_6j(l, j, s, jdash, l, 1)

    def G3(self, chi, chj):

        j = chi.j
        jdash = chj.j
        l = chi.l
        s = chi.s

        return (-1) ** (s + l + jdash + 1) * np.sqrt((2 * j + 1) * (2 * jdash + 1)) * wigner_6j(s, j, l, jdash, s, 1)

    def G4(self, chi, chj, Ft, Ftdash):
        Fc = chi.core.f
        Fcdash = chj.core.f
        j = chi.j
        jdash = chj.j

        if j == jdash:
            return (-1) ** (Fc + j + Ftdash + 1) * np.sqrt((2 * Ft + 1) * (2 * Ftdash + 1)) * wigner_6j(Fc, Ft, j, Ftdash, Fcdash,1)
        else:
            return 0

    def G5(self, chi, chj):
        Jc = chi.core.j
        Jcdash = chj.core.j
        Fc = chi.core.f
        Fcdash = chj.core.f
        I = chi.core.i

        if Jc == Jcdash:
            return (-1) ** (Jc + I + Fc + 1) * np.sqrt((2 * Fc + 1) * (2 * Fcdash + 1)) * wigner_6j(I, Fc, Jc, Fcdash, I, 1)
        else:
            return 0

    def G6(self, chi, chj):
        Jc = chi.core.j
        Jcdash = chj.core.j
        Fc = chi.core.f
        Fcdash = chj.core.f
        I = chi.core.i

        return (-1) ** (Jc + I + Fcdash + 1) * np.sqrt((2 * Fc + 1) * (2 * Fcdash + 1)) * wigner_6j(Jc, Fc, I, Fcdash, Jcdash, 1)

    def G7(self, chi, chj):
        Jc = chi.core.j
        Jcdash = chj.core.j
        Sc = chi.core.s
        Lc = chi.core.l

        return (-1) ** (Sc + Lc + Jc + 1) * np.sqrt((2 * Jc + 1) * (2 * Jcdash + 1)) * wigner_6j(Lc, Jc, Sc, Jcdash, Lc,1)

    def G8(self, chi, chj):
        Jc = chi.core.j
        Jcdash = chj.core.j
        Sc = chi.core.s
        Lc = chi.core.l

        return (-1) ** (Sc + Lc + Jcdash + 1) * np.sqrt((2 * Jc + 1) * (2 * Jcdash + 1)) * wigner_6j(Sc, Jc, Lc, Jcdash, Sc, 1)
    
    # these functions have some naming collisions with numerov--FIXME

    # def polarizability(self,lam_nm,transitions):
    #     omega = 2*np.pi*cs.c/(lam_nm*1e-9)
        
    #     j = transitions[0].ji
    #     k = 0
        
    #     prefactor = (-1)**(j + k + 1)
        
    #     def pre(t):
    #         return (-1)**t.jf * wigner_6j(1,k,1,j,t.jf,j) * 3*np.pi*cs.epsilon_0 *cs.c**3 * (2*t.jf+1) * 1e9/(t.lifetime_ns) / t.omega_sec**3
        
    #     def det(t,omega):
    #         return 1/(t.omega_sec - omega - 1.j*1e9/(2*t.lifetime_ns)) + (-1)**k/(t.omega_sec + omega + 1.j*1e9/(2*t.lifetime_ns))
        
    #     contributions = [prefactor * pre(t) * det(t,omega) for t in transitions]
  
    #     alpha_0 = np.sum(contributions,axis=0)
    #     alpha_s = alpha_0 / np.sqrt(3*(2*j+1))
        
    #     return alpha_s

    # def scattering_rate(self,lam_nm,transitions):
    #     # returns scattering rate in Hz/(W/m^2)
        
    #     alpha = self.polarizability(lam_nm,transitions=transitions)
    #     omega = 2*np.pi*cs.c/(lam_nm*1e-9)
        
    #     Esq = 2/(cs.epsilon_0*cs.c)
        
    #     return 0.5*omega*np.imag(alpha)*Esq / (cs.hbar * omega)
    
    # def potential(self,lam_nm,transitions):
    #     # returns potential in Hz/(W/m^2)
        
    #     alpha = self.polarizability(lam_nm,transitions=transitions)
    
    #     Esq = 2/(cs.epsilon_0*cs.c)
        
    #     return -0.25*np.real(alpha)*Esq / cs.h

    # def scattering_rate_neutral(self,lam_nm):
    #     return self.scattering_rate(lam_nm,transitions=self.transitions_neutral)
    
    # def potential_neutral(self,lam_nm):
    #     return self.potential(lam_nm,transitions=self.transitions_neutral)
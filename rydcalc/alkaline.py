import numpy as np
import sqlite3

from .alkali import *


class AlkalineAtom(AlkaliAtom):
    
    transitions_ion = []
    
    def potential_ion(self,lam_nm):
        return self.potential(lam_nm,transitions=self.transitions_ion)
    
    def scattering_rate_ion(self,lam_nm):
        return self.scattering_rate(lam_nm,transitions=self.transitions_ion)
    
    def repr_state(self,st):
        return st.pretty_str
    
        # """ generate a nice ket for printing """
        # if st.tt == 'nljm': 
        #     if st.l <= 5:
        #         return "|%s:%d,%d,%s,%.1f,%.1f>" % (self.name,st.n,2*st.s+1,['S','P','D','F','G','H'][st.l],st.j,st.m)
        #     else:
        #         return "|%s:%d,%d,%d,%.1f,%.1f>" % (self.name,st.n,2*st.s+1,st.l,st.j,st.m)
        # if st.tt == 'composite':
        #     out = ""
        #     #FIXME: trailing + sign
        #     for p,c in zip(st.parts,st.wf_coeffs):
        #         out += "%.2e %s + " % (c,p.__repr__())
            
        #     return out
        
    # FIXME: put in proper Lande g factor
    # def get_g(self,st):
    #     if st.tt == 'composite':
    #         g = 0
    #         for p,c in zip(st.parts,st.wf_coeffs):
    #             g += np.abs(c)**2 * p.get_g()
    #     else:
    #         g = super().get_g(st)
        
    #     return g
    
    def get_state(self,qn,tt='nsljm'):
        # first, find suitable channel

        if tt == 'nsljm' and len(qn)==5:
            
            n = qn[0]
            s = qn[1]
            l = qn[2]
            j = qn[3]
            m = qn[4]
            
            # for defect model
            qns = {'n': n, 's':s, 'l': l, 'j': j}
            
            if s < 0 or l < 0 or l >= n or np.abs(m) > j or j < np.abs(l-s) or j > l+s:
                return None
            
            if l <= 5:
                pretty_str = "|%s:%d,%d,%s,%.1f,%.1f>" % (self.name,n,2*s+1,['S','P','D','F','G','H'][l],j,m)
            else:
                pretty_str = "|%s:%d,%d,%d,%.1f,%.1f>" % (self.name,n,2*s+1,l,j,m)
            
        else:
            print("tt=",tt," not supported by H.get_state")
        
        my_ch = None
        
        # now decide what our core channels should be
        # here, we take core electron to be in S=1/2, L=1/2, J=1/2 state
        # so we really just have to decide on j of Rydberg electron
        
        sr = 1/2
        lr = l
        coeffs = [1]
        
        if s==1 and j == l+1:
            # ie, 3P2
            jr = [l + 1/2]
        elif s==1 and j == l-1:
            # ie, 3P0
            jr = [l - 1/2]
        else:
            # ie, 3P1 or 1P1
            jr = [l+1/2,l-1/2]
            
            theta = self.get_spin_mixing(l)
            
            rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            
            ls_to_jj = np.array([[np.sqrt(l/(2*l+1)), -np.sqrt((l+1)/(2*l+1))], [np.sqrt((l+1)/(2*l+1)), np.sqrt(l/(2*l+1))]])
            
            ls_to_jj_rot = rot@ls_to_jj
            
            # # Eq. 4 in Lurio 1962 [had to change sign to make it work... try to figure out why this is]
            # if s==1:
            #     coeffs = [np.sqrt(l/(2*l+1)), -np.sqrt((l+1)/(2*l+1))]
            # else:
            #     coeffs = [np.sqrt((l+1)/(2*l+1)), np.sqrt(l/(2*l+1))]
            
            if s==1:
                coeffs = ls_to_jj_rot[0]
            else:
                coeffs = ls_to_jj_rot[1]

        chs = []
        
        for ij,ic in zip(jr,coeffs):
            
            my_ch = None
            
            # search through existing channels
            for ch in self.channels:
                if ch.l == lr and ch.j == ij and ch.s == sr:
                    my_ch = ch
                    break
            
            # if we didn't find a channel, make a new one
            if my_ch is None:
                my_ch = channel(self.core,(sr,lr,ij),tt='slj')
                self.channels.append(my_ch)
                
            chs.append(my_ch)
            
        defect = self.get_quantum_defect(qns)
        energy_Hz = self.get_energy_Hz_from_defect(n,defect)
            
        #__init__(self,atom,qn,channel,energy = None,tt='npfm'):
        st = state_mqdt(self,(n,(-1)**l,j,m),coeffs,chs,energy_Hz = energy_Hz)
        st.pretty_str = pretty_str
        return st

    def get_nearby(self,st,include_opts={}):
        """ generate a list of quantum number tuples specifying nearby states for sb.fill().
        include_opts can override options in terms of what states are included.
        
        It's a little messy to decide which options should be handled here vs. in single_basis
        deecision for now is to have all quantum numbers here but selection rules/energy cuts
        in single_basis to avoid duplication of code.
        
        Does not create states or check valid qn, just returns list of tuples. """
        
        ret = []
        
        o = {'dn': 2, 'dl': 2, 'dm': 1, 'ds': 0}
        
        for k,v in include_opts.items():
            o[k] = v
            
        for n in np.arange(st.n-o['dn'],st.n+o['dn']+1):
            for s in np.arange(max(0,st.channels[0].s-o['ds']), st.channels[0].s+o['ds']+1):
                for l in np.arange(st.channels[0].l-o['dl'],st.channels[0].l+o['dl']+1):
                    for j in np.arange(st.f-o['dl'],st.f+o['dl']+1):
                        for m in np.arange(st.m-o['dm'],st.m+o['dm']+1):
                        
                            ret.append((n,s,l,j,m))
        
        return ret
    
    def get_magnetic_me(self, st, other, cutoffnu=10):
        """ Return the magnetic dipole matrix element between st and other.

        Calculated using Eq. 21 of Robichaux et al, 10.1103/PhysRevA.97.022508
        """

        def lam(x):
            return np.sqrt((2 * x + 1) * (x + 1) * x)

        reduced_me = 0
        Ft = st.f
        Ftdash = other.f

        muB = cs.physical_constants['Bohr magneton'][0]
        muN = cs.physical_constants['nuclear magneton'][0]
        gs = -cs.physical_constants['electron g factor'][0]
        muI = self.gI * muN

        if st.m != other.m or np.abs(st.f - other.f) > 1 or st.parity != other.parity:
            return 0

        prefactor = (-1) ** (st.f - st.m) * wigner_3j(st.f, 1, other.f, -st.m, 0, other.m)

        for ii, (Ai, chi) in enumerate(zip(st.Ai, st.channels)):
            for jj, (Aj, chj) in enumerate(zip(other.Ai, other.channels)):

                chinu = 1 / np.sqrt((chi.core.Ei_Hz - st.energy_Hz) / (self.RydConstHz))
                chjnu = 1 / np.sqrt((chj.core.Ei_Hz - other.energy_Hz) / (self.RydConstHz))

                if np.abs(chinu - chjnu) > cutoffnu:  # cut-off for small overlap
                    continue

                if chi.no_me or chj.no_me:
                    continue

                if chi.core.l < 0 or chi.l != chj.l or chi.s != chj.s or chi.core.i != chj.core.i or chi.core.l != chj.core.l or chi.core.s != chj.core.s:
                    # chi.core.l>=0 to exclude unknown effective core states. implemented with l=-1
                    continue

                ll = self.G1(chi, chj, Ft, Ftdash) * self.G2(chi, chj) * lam(chi.l)
                ss = self.G1(chi, chj, Ft, Ftdash) * self.G3(chi, chj) * lam(chi.s)
                II = self.G4(chi, chj, Ft, Ftdash) * self.G5(chi, chj) * lam(chi.core.i)
                LL = self.G4(chi, chj, Ft, Ftdash) * self.G6(chi, chj) * self.G7(chi, chj) * lam(chi.core.l)
                SS = self.G4(chi, chj, Ft, Ftdash) * self.G6(chi, chj) * self.G8(chi, chj) * lam(chi.core.s)

                if chinu == chjnu:
                    overlap = 1
                else:
                    overlap = (2 * np.sqrt(chinu * chjnu) / (chinu + chjnu)) * (
                                np.sin(np.pi * (chinu - chi.l) - np.pi * (chjnu - chj.l)) / (
                                    np.pi * (chinu - chi.l) - np.pi * (chjnu - chj.l)))

                reduced_me += overlap * Ai * np.conjugate(Aj) * (muB * (LL + ll + gs * (SS + ss)) - muI * II)

                # need to implement other q for coupling
                # mu+=overlap*Ai*np.conjugate(Aj)*np.conjugate(((-1)**(st.f-st.m))*wigner_3j(st.f,1,other.f,-st.m,0,other.m))*(muB*(LL+ll+gs*(SS+ss))-muI*II)

        me = prefactor * reduced_me
        return me / muB
    
    def _dipole_db_query(self,s1,s2,rwi,me=0):
        
        # Provide strings to query to/from dipole matrix element database.
        # Ideally, this is agnostic about which type of matrix element is being stored,
        # and is the only thing that needs to be modified for different types of atoms
        # with different quantum numbers.
        
        # if we re-imported ARC databases with spin, we could probably get away
        # without this since all (conceivably interesting) atoms could be described
        # by nslj
        
        # Given the need to have different strings for creating tables and loading
        # from files, this function has become a bit ugly
        
        # length of this should be 2*(number of quantum numbers) + 1 for matrix element
        insert_str = "(?,?,?,?,?,?,?,?,?)"
        
        if rwi=='i':
            # query for initializing database
            query_str = '''(n1 TINYINT UNSIGNED, s1 TINYINT UNSIGNED, l1 TINYINT UNSIGNED,
                 j1_x2 TINYINT UNSIGNED,
                 n2 TINYINT UNSIGNED, s2 TINYINT UNSIGNED, l2 TINYINT UNSIGNED,
                 j2_x2 TINYINT UNSIGNED,
                 dme DOUBLE,
                 PRIMARY KEY (n1,s1,l1,j1_x2,n2,s2,l2,j2_x2 )
                 )'''
            
            return query_str
        
        if rwi=='wf':
            # query for writing to db from file
            query_str = insert_str
            
            return query_str
        
        if rwi=='rf':
            # query for read from db to save to file
            query_str = 'n1,s1,l1,j1_x2,n2,s2,l2,j2_x2'
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
            query_str = "n1= ? AND s1 = ? AND l1 = ? AND j1_x2 = ? AND n2 = ? AND s2 = ? AND l2 = ? AND j2_x2 = ?"
            query_dat = (s1_o.n, s1_o.s, s1_o.l, int(2*s1_o.j), s2_o.n, s2_o.s, s2_o.l, int(2*s2_o.j))
        
        if rwi=='w':
            # query for writing to database (storing matrix element me)
            query_str = insert_str
            query_dat = (s1_o.n, s1_o.s, s1_o.l, int(2*s1_o.j), s2_o.n, s2_o.s, s2_o.l, int(2*s2_o.j),me)
        
        return query_str, query_dat


    

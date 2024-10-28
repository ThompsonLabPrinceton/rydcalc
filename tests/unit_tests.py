# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import rydcalc
import unittest
import numpy as np
import scipy.special
import scipy.constants as cs

class TestHydrogen(unittest.TestCase):
    
    def setUp(self):
        
        self.H = rydcalc.Hydrogen()
        self.Rb = rydcalc.Rubidium87()
        
    def test_norms(self):
        
        max_dev = 0
        
        for n in np.arange(10,150,10):
            
            st = self.H.get_state((n,0,0))
            
            me = self.H.radial_integral(st,st,overlap=True)
            
            self.assertAlmostEqual(np.abs(me),1,places=7,msg="n=%d" % n)
            
            if np.abs(me)-1 > max_dev:
                max_dev = np.abs(me) - 1
                
        print("Passed wavefunction norm test for Hydrogen (max deviation=%.2e)." % max_dev)
        
    def test_multipole_circ(self):
        """ Two notes about this one:
            
            1. It does not pass for n > 75, which is where H.radial_wavefunction switches to computing
            log(R). The answers are still very close (1%), but it may be worth looking into this.
            
            2. I am also not sure if the reduced mass correction makes sense.
            """
        
        max_dev = 0
        
        at = self.H
        
        for k in [1,2,3]:
            for n in np.arange(10,80,10):
                
                st = at.get_state((n,n-1,n-1))
                st2 = at.get_state((n-k,n-1-k,n-1-k))
                
                me = at.get_multipole_me(st,st2,k=k)#*self.H.mu**k
                me_th = circ_me(n,k,at.mu)
                
                #try:
                #self.assertAlmostEqual(me,me_th,places=2)
                self.assertAlmostEqual(1,me/me_th,places=7,msg="(n,k)=(%d,%d)" % (n,k))
                #except:
                #    print("Failed on n,k = ",n,k)
                
                if np.abs(me-me_th) > max_dev:
                    max_dev = np.abs(me - me_th)/me_th
                
        print("Passed wavefunction matrix element test for Hydrogen (max deviation=%.2e)." % max_dev)
        
    # def test_multipole_lowl(self):
    #     """ The wavefunctions for some multipole transitions from nS -> (n+1)k are computed
    #     and compared to analyitic results from Mathematica.
        
    #     There is not a simple closed-form expression for the general matrix element. """
        
    #     ans = []
        
    #     ans.append({'k': 1, 'n': 1, 'me': 128*np.sqrt(2)/243})
    #     ans.append({'k': 1, 'n': 5, 'me': 5914803600000*np.sqrt(14)/3138428376721})
        
    #     ans.append({'k': 2, 'n': 1, 'me': 81*np.sqrt(3/2)/128})
    #     ans.append({'k': 2, 'n': 5, 'me': -1698835053125*np.sqrt(35/3)/52242776064})
        
    #     ans.append({'k': 3, 'n': 1, 'me': 786432/np.sqrt(5)/390625})
    #     ans.append({'k': 3, 'n': 5, 'me': -48746899046400000000*np.sqrt(330)/665416609183179841})
        
    #     for a in ans:
                
    #         st = self.H.get_state((a['n'],0,0))
    #         st2 = self.H.get_state((a['n']+a['k'],a['k'],a['k']))
            
    #         me = self.H.get_multipole_me(st,st2,k=a['k'])*self.H.mu**a['k']
            
    #         me_th = a['me']
            
    #         #self.assertAlmostEqual(me,me_th,places=2)
    #         self.assertAlmostEqual(1,me/me_th,places=7)
            
    # def test_circ_lifetime(self):
        
    #     """ Test the zero-temperature lifetime of several circular states.
        
    #     Fails above n=75, with log wavefunctions in H.radial_wavefunction() """
        
    #     for n in np.arange(10,80,10):
                
    #         st = self.H.get_state((n,n-1,n-1))
            
    #         env = rydcalc.environment(T_K=0)

    #         lifetime = 1/self.H.total_decay(st,env)*self.H.mu
                
    #         #try:#
    #         self.assertAlmostEqual(1,lifetime/circ_lifetime(n,0),places=6)
    #         #except:
    #         #    print(n,t)
                
    # def test_circ_lifetime_finite(self):
        
    #     """ Test the finite-temperature partial decay rate to the next lowest circular state.
        
    #     The agreement here is not as good for reasons that are unclear, so we set places=2 to check for gross errors.
    #     """
        
    #     for n in np.arange(10,80,10):
            
    #         for t in [4,10,100,300]:
                
    #             st = self.H.get_state((n,n-1,n-1))
    #             st2 = self.H.get_state((n-1,n-2,n-2))
                
    #             env = rydcalc.environment(T_K=t)

    #             lifetime = 1/self.H.partial_decay(st,st2,env)*self.H.mu
                    
    #             #print(lifetime,circ_lifetime(n,t))
                    
    #             #try:#
    #             self.assertAlmostEqual(1,lifetime/circ_lifetime(n,t),places=2)
    #             #except:
    #             #    print(n,t)
                
            
          
def circ_me(n,k,mu=1):
    """ Analytic multipole matrix element from |n,n-1,n-1> down to |n-k,n-1-k,n-1-k>.
    
    Compute the log to ensure we are able to evaluate for large n, k.
    
    Note that these matrix elements are computed in units of a_0 involving reduced mass, so we have
    to scale by that.
    """
    
    gln = scipy.special.gammaln
    
    log_me = 0
    
    log_me += (2*n+1)*np.log(2) + np.log((-1)**k*(k-n)**k) + np.log(n) - n*np.log(n*(n-k))
    log_me += -2*n*np.log(1/n+1/(n-k))+0.5*gln(k+0.5) - 0.5*gln(k+1) +gln(n)
    log_me += -np.log(np.pi**0.25 * (2*n-k)) - gln(n-k)
    
    return (-1)**k * (1/mu)**k * np.exp(log_me)
    

def circ_lifetime(n,t):
    """ Analytic expression for circular state lifetime vs. temp. Adapted from Eq. 5,6 in
    
    Xia, Zhang and Saffman, PHYSICAL REVIEW A 88, 062337 (2013)
    
    """
    
    prefactor = np.pi**5 * cs.c**3 * cs.epsilon_0**5 * cs.hbar**6 / (cs.m_e * cs.e**10)
    
    n_factor_log = np.log(3) + (5-2*n)*np.log(4) + (4*n-1)*np.log(2*n-1) - (2*n-2)*np.log(n-1) - (2*n - 4)*np.log(n)
    
    dE = -cs.Rydberg * cs.c * (1/(n**2) - 1/((n-1)**2)) * cs.h
    
    if t > 0:
        temp_factor = (1/(np.exp(dE/(cs.Boltzmann*t)) - 1) +1)
    else:
        temp_factor = 1
    
    return prefactor*np.exp(n_factor_log)/temp_factor

if __name__ == '__main__':
    unittest.main(verbosity=2)

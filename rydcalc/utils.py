
from sympy.physics.wigner import wigner_3j as wigner_3j_sympy
from sympy.physics.wigner import wigner_6j as wigner_6j_sympy
from sympy.physics.wigner import wigner_9j as wigner_9j_sympy
from sympy.physics.quantum.cg import CG as CG_sympy

import numpy as np

import functools, copy

# some functions for wigner symbols
# this turns out to be a lot faster than a sqlite db for the relevant cache sizes
@functools.lru_cache(maxsize=256)
def wigner_3j(j1,j2,j3,m1,m2,m3):
    return float(wigner_3j_sympy(j1,j2,j3,m1,m2,m3))

@functools.lru_cache(maxsize=256)
def wigner_6j(j1,j2,j3,j4,j5,j6):
    try:
        ans = float(wigner_6j_sympy(j1,j2,j3,j4,j5,j6))
        return ans
    except:
        print("Error in Wigner6j, with input ",(j1,j2,j3,j4,j5,j6))
        raise ValueError
        
@functools.lru_cache(maxsize=256)
def wigner_9j(j1,j2,j3,j4,j5,j6,j7,j8,j9):
    try:
        ans = float(wigner_9j_sympy(j1,j2,j3,j4,j5,j6,j7,j8,j9))
        return ans
    except:
        print("Error in Wigner9j, with input ",(j1,j2,j3,j4,j5,j6,j7,j8,j9))
        raise ValueError
    
@functools.lru_cache(maxsize=256)
def CG(*args):
    return float(CG_sympy(*args).doit().evalf())


class model_params:
    
    def __init__(self,values=None):
        """ Initialize the model params. If we don't pass a dictionary, it will return default values for
        value(). If we do, it will override with values from dictionary. """
    
        if values is None:
            
            self.values = {}
            self.sigmas = {}
            self.initialized = False
            
        else:
            
            self.values = copy.deepcopy(values)
            self.sigmas = {}
            self.initialized = True
            
        self.set_prefix('')
            
    def set_prefix(self,prefix):
        self.prefix = prefix
        
    def value(self,name,value=None,sigma=None):
        """ Get value[name] from params dictionary. If we have not initialiezd this class with a dictionary of values,
        the value argument is returned as a default. """
        
        if self.prefix == '':
            path = name
        else:
            path = self.prefix + '_' + name
        
        if sigma is None:
            self.sigmas[path] = value/1000
        else:
            self.sigmas[path] = sigma
            
        if self.values.get(path) is None:
            
            self.values[path] = value
            return value
        
        else:
            
            return self.values.get(path)
        
    def get_vector(self,keys = None):
        
        if keys == None:
        
            return np.array(list(self.values.values()))
        
        else:
            
            return np.array([self.values[k] for k in keys])

    def get_sigmas(self, keys = None):
        
        if keys == None:
            
            return np.array(list(self.sigmas.values()))
        
        else:
            
            return np.array([self.sigmas[k] for k in keys])



        
        
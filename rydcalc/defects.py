import numpy as np
import scipy as sp
import scipy.integrate
import scipy.interpolate
import scipy.constants as cs
import scipy.optimize
# whether to silence warnings about validity of quantum defect ranges
qd_quiet = False

class defect_model:
    ''' Parent class for quantum defect model. Has the following methods:

    - is_valid(qns)
        Checks whether the defect model is valid for the given quantum numbers (returns True/False).

    - get_defect(qns)
        Returns the defect for the given quantum numbers.

    This model is intended to be subclassed by other classes, such as a Rydberg-Ritz model or a model based on experimental energy levels.
    
    The corrections options specify whether various corrections to the energy, beyond the quantum defect, should be included.

    The intended usage of defect_model is to populate the atomic class (ie, Rubidium87) with a list of models, which
    will be sequentially checked to determine the first one that is valid for a given state.
    '''
    
    def __init__(self, delta, condition=lambda qns: True, polcorrection=False, SOcorrection=False, relcorrection=False):
        """
        Initializes an instance of a quantum defect model.

        Args:
            delta (float): Quantum defect
            condition (function, optional): A lambda function taking quantum numbers as input and returning True or False to determine whether this defect model is valid for the given state. Defaults to lambda qns: True.
            polcorrection (bool, optional): Flag indicating whether polarization correction is applied. Defaults to False.
            SOcorrection (bool, optional): Flag indicating whether spin-orbit correction is applied. Defaults to False.
            relcorrection (bool, optional): Flag indicating whether relativistic correction is applied. Defaults to False.
        """
        self.delta = delta
        self.condition = condition
        self.corrections = {"polcorrection": polcorrection, "SOcorrection": SOcorrection, "relcorrection": relcorrection}

    def is_valid(self,qns):
        ''' test if this defect model is valid for this state '''
        return self.condition(qns)
    
    def get_defect(self,qns):
        return self.delta

class defect_Rydberg_Ritz(defect_model):
    '''Subclass of defect_model implementing the modified Rydberg-Ritz formula,
    
    delta = delta_0 + delta_2 / (n-delta_0)^2 + delta_4 / (n-delta_0)^4 ... '''
    
    # it is a little unclear whether we want to have both condition function
    # and n_range. The idea of n_range is that it is a 'soft' error as opposed
    # to throwing a zero by default
    
    def __init__(self,deltas,n_range = None,condition=lambda qns: True, polcorrection= False, SOcorrection= False, relcorrection= False):
        self.deltas = deltas
        self.order = len(self.deltas)
        self.n_range = n_range
        self.condition=condition
        self.corrections = {"polcorrection": polcorrection,"SOcorrection":SOcorrection,"relcorrection":relcorrection}

        if self.order == 0:
            print("Error--tried to initialize empty defect_Rydberg_Ritz")
    
    def get_defect(self,qns):
        
        if self.n_range is not None and (self.n_range[0] > qns['n'] or self.n_range[1] < qns['n']) and not qd_quiet:
            print("Warning--using defect_Rydberg_Ritz outside of specified n_range for st: ",qns,"(range is ",self.n_range,")")
        
        defect = self.deltas[0]
        
        nord = 1
        while nord < self.order:
            defect += self.deltas[nord] / (qns['n'] - self.deltas[0])**(2*nord)
            nord+=1
        
        return defect
    
class defect_from_energy(defect_model):
    """ Class to define quantum defect based on experimentally known energy. This uses the reduced mass of the atom so that the correct energy will be returned when converting back to Hz. """
    
    def __init__(self,energy,unit='Hz',condition=lambda qns: True):
        """
        Initialize the defect_from_energy class.

        Parameters:
        - energy: The experimentally known energy, relative to the threshold.
        - unit: The unit of the energy, '1/cm' or 'Hz' (default is 'Hz').
        - condition: A lambda function that defines a condition for the defect calculation (default is lambda qns: True).
        """
        
        if unit == 'Hz':
            self.energy_Hz = energy
        
        elif unit == '1/cm':
            self.energy_Hz = energy * cs.c * 100
        
        else:
            print("Error--unsupported unit", unit, 'in defect_from_energy')
        
        self.energy_au = self.energy_Hz / (2 * cs.Rydberg * cs.c)
        self.condition=condition
        
    def get_defect(self,qns):

        return qns['n'] - np.sqrt(-1/(2*self.energy_au))
        
    
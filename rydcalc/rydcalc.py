#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:29:18 2020

@author: jt11
"""
#test
import numpy as np
#from sympy.physics.wigner import wigner_3j,wigner_6j,wigner_9j
# from sympy.physics.wigner import wigner_3j as wigner_3j_sympy
# from sympy.physics.wigner import wigner_6j as wigner_6j_sympy
# from sympy.physics.wigner import wigner_9j
# from sympy.physics.hydrogen import R_nl
# from sympy.physics.quantum.cg import CG as CG_sympy

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

#a0 = cs.physical_constants['Bohr radius'][0]

# from .ponderomotive import *
# from .db_manager import *
# from .constants import *

# from .hydrogen import *
# from .alkali import *
# from .alkaline import *

# from .alkali_data import *
# from .alkaline_data import*

# from .utils import *


class environment:
    """ class that specifies environmental parameters, including E, B, and DOS """
    
    def __init__(self,Bz_Gauss=0,Ez_Vcm=0,T_K=0,Intensity_Wm2=0,diamagnetism=False):
        
        self.Bz_Gauss = Bz_Gauss # Z-directed B-field in Gauss
        self.Ez_Vcm = Ez_Vcm     #Z-directed E-field in V/cm
        self.T_K = T_K        # temperature in Kelvin
        
        self.Intensity_Wm2 = Intensity_Wm2 # Peak light intensity in W/m^2 for ponderomotive potential
        
        # this should take freq. in Hz and polarization +/-1, 0 and return
        # normalized LDOS (ie, Purcell factor)
        self.ldos = lambda f,q: 1

        # diamagnetism is a flag to include the diamagnetic term in the Hamiltonian
        self.diamagnetism = diamagnetism
        
        # potential addition: two-point DOS to describe interactions in structured environment
        
    def __repr__(self):
        return "Bz=%.2f G, Ez=%.2f V/cm, T=%.2f K" % (self.Bz_Gauss, self.Ez_Vcm, self.T_K)





def getCs(pb,env,th=np.pi/2,phi=0,rList_um=np.logspace(0.3,2,20),plot=True):
    ''' Compute C6 and C3 coefficients for the highlighted pairs in pb.
    
    Returns [C6d,C6e,C3d,C3e] where e denotes exchange (off-diagonal) interation
    and d is diagonal interaction.
    
    In computing these, only looks at first two highlighted pairs, and takes
    them to be in the order [g,u] as implemented in pb.fill()
    
    '''
    
    Nb = pb.dim()

    rumList = rList_um

    energies = []
    energiesAll = []
    overlaps = []

    en0 = pb.computeHtot(env,0,th=th,phi=phi,interactions=False)[0,0]

    for rum in rumList:
        
        ret = pb.computeHtot(env,rum,th=th,phi=phi,interactions=True)

        energiesAll.append(pb.es)

        energies.append(ret[:,0]-en0)
        overlaps.append(ret[:,1])

    energies = np.array(energies)
    overlaps = np.array(overlaps)
    energiesAll = np.array(energiesAll)
    
    #return energies
    
    def intfn(r,c6,c3):
        return c6/r**6 + c3/r**3

    # rMinFit = 5
    # eMaxFit_Hz = 1e9
    # rFitIdx = np.argwhere(rumList > rMinFit).flatten()
    # eFitIdx = np.argwhere(np.abs(energies) < eMaxFit_Hz).flatten() # sort of arbitrary, should probably scale with n
    # fitIdx = np.intersect1d (rFitIdx,eFitIdx)        


    #return (rumList[fitIdx],np.real(energies[fitIdx,0]))
    interactionFits = []
    
    for ii in range(len(pb.highlight)):
        popt,pcov = sp.optimize.curve_fit(intfn,rumList,np.real(energies[:,ii]),p0=(1e9,1e7))
        # fitting on a log scale does better when the r range is large, although it's not clear
        #popt,pcov = sp.optimize.curve_fit(lambda r,c6,c3: np.log(intfn(np.exp(r),c6,c3)),np.log(rumList[:]),np.log(np.real(energies[:,ii])),p0=(1e9,1e7))
        interactionFits.append(popt)
    
    if len(interactionFits) > 1:
        c6d = (interactionFits[0][0] + interactionFits[1][0])/2
        c6e = (interactionFits[0][0] - interactionFits[1][0])/2
        c3d = (interactionFits[0][1] + interactionFits[1][1])/2
        c3e = (interactionFits[0][1] - interactionFits[1][1])/2
    else:
        c6d = interactionFits[0][0]
        c6e = 0
        c3d = interactionFits[0][1]
        c3e = 0

    if plot:
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)

        for ii in range(len(pb.highlight)):
            posidx = np.argwhere(energies[:,ii] >=0)
            negidx = np.argwhere(energies[:,ii] < 0)
            plt.plot(rumList[posidx],np.abs(energies[posidx,ii])*1e-6,'^',color='C'+str(ii),label=repr(pb.highlight[ii][:2]))
            plt.plot(rumList[negidx],np.abs(energies[negidx,ii])*1e-6,'v',color='C'+str(ii))
            plt.plot(rumList,np.abs(intfn(rumList,*interactionFits[ii]))*1e-6,'-',color='C'+str(ii))

        plt.xlabel('r [um]')
        plt.ylabel('Pair Energy [MHz]')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim([-10,10])
        plt.grid(axis='both')

        plt.subplot(1,2,2)

        for ii in range(len(pb.highlight)):
            plt.plot(rumList,overlaps[:,ii],'-',label=repr(pb.highlight[ii][:1]))

        plt.xlabel('r [um]')
        plt.ylabel('Overlap')
        #plt.legend()
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylim([-0.1,1.2])
        plt.grid(axis='both')
        
    return c6d,c6e,c3d,c3e
    

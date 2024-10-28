#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:21:17 2020

@author: jt11

This file should have analysis functions that give actual answers about states,
and expose the inner workings of the calculations as little as possible.

However, we make them all classes so the underlying data is still available for
further introspection
"""

#from rydcalc import *
from .single_basis import *
from .pair_basis import *

from rydcalc import * # to get environment

#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from tqdm import tqdm

class analysis:
    pass
        

class analysis_stark(analysis):
    """ Example class for performing an analysis of the Stark shift. This function constructs
    a basis of states, then diagonalizes the Hamiltonian for a given E and B field, and makes a plot.
    """
    def __init__(self,s1,include_opts={}):
        """
        Initialize the analysis_stark class with a state object and optional parameters.

        Parameters:
        s1 (state): The state to be analyzed.
        include_opts (dict): Optional parameters to include in the analysis, such as dn, dl, dm, and dipole_allowed.

        The function sets up the basis from the MQDT model, computes the Hamiltonian, and prepares for the Stark shift analysis.
        """
        
        self.opts = {'dn': 5,'dl': 5,'dm': 5,'dipole_allowed': False}
        
        for k,v in include_opts.items():
            self.opts[k] = v
        
        self.sb = single_basis()
        self.sb.fill(s1,include_opts = self.opts)
        self.sb.compute_hamiltonian()
        
    def quad_fit(self,e,lin,quad):
        return e*lin + e**2 * quad

    def pure_quad_fit(self,e,quad):
        return e**2 * quad
        
    def run(self,Bz_Gauss=0,Ez_list_Vcm = np.arange(0,1,0.1),plot=False,diamagnetism = False):
        """
        Run the Stark shift analysis for a range of electric fields.

        Parameters:
        Bz_Gauss (float): The magnetic field in Gauss.
        Ez_list_Vcm (np.array): List of electric fields in V/cm for which the analysis is performed [default: np.arange(0,1,0.1)]
        plot (bool): If True, plots the Stark shift results [default: False]

        This method computes the energies and overlaps for each electric field in Ez_list_Vcm,
        fits the quadratic Stark coefficient, and optionally plots the results.
        """
        
        self.env = environment(Bz_Gauss=Bz_Gauss,Ez_Vcm = 0, diamagnetism = diamagnetism)
        self.Ez_list_Vcm = Ez_list_Vcm
        
        self.energies = []
        self.energiesAll = []
        self.overlaps = []
        
        self.en0All = self.sb.compute_energies(self.env)
        self.en0 = self.en0All[0,0]
        
        
        for ez in self.Ez_list_Vcm:
            
            self.env.Ez_Vcm = ez
            
            ret = self.sb.compute_energies(self.env)
    
            self.energiesAll.append(self.sb.es - self.en0)
    
            self.energies.append(ret[:,0] - self.en0)
            self.overlaps.append(ret[:,1])
    
        self.energies = np.real(np.array(self.energies))
        self.overlaps = np.real(np.array(self.overlaps))
        self.energiesAll = np.real(np.array(self.energiesAll))
        
        self.starkFits = []
        
        for ii in range(len(self.sb.highlight)):
            p0quad = self.energies[-1,ii]/self.Ez_list_Vcm[-1]**2
            popt,pcov = sp.optimize.curve_fit(self.quad_fit,self.Ez_list_Vcm,self.energies[:,ii],p0=(0,p0quad))
            # fitting on a log scale does better when the r range is large, although it's not clear
            #popt,pcov = sp.optimize.curve_fit(lambda r,c6,c3: np.log(intfn(np.exp(r),c6,c3)),np.log(rumList[:]),np.log(np.real(energies[:,ii])),p0=(1e9,1e7))
            self.starkFits.append(popt)
        
        if plot:
            
            xlabel = 'Electric Field [V/cm]'
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
    
            for ii in range(len(self.sb.highlight)):
                plt.plot(self.Ez_list_Vcm,self.energies[:,ii]*1e-6,'v',color='C'+str(ii),label=repr(self.sb.highlight[ii][:1]))
                plt.plot(self.Ez_list_Vcm,self.quad_fit(self.Ez_list_Vcm,*self.starkFits[ii])*1e-6,'-',color='C'+str(ii))
    
            plt.xlabel(xlabel)
            plt.ylabel('Energy [MHz]')
            plt.legend()
            #plt.ylim([-10,10])
            plt.grid(axis='both')
    
            plt.subplot(1,2,2)
    
            for ii in range(len(self.sb.highlight)):
                plt.plot(self.Ez_list_Vcm,self.overlaps[:,ii],'-',label=repr(self.sb.highlight[ii][:1]))
    
            plt.xlabel(xlabel)
            plt.ylabel('Overlap')
            #plt.legend()
            #plt.yscale('log')
            plt.ylim([-0.1,1.2])
            plt.grid(axis='both')
        
        return self.starkFits
    
    def plot_stark_map(self,energy_range_Hz = [-1e9,1e9]):
        """
        Plots the Stark map for the system within a specified energy range.

        Args:
            energy_range_Hz (list): A two-element list specifying the minimum and maximum energy range in Hz for the plot.

        Returns:

            fig,ax for further modification.
        """

        fig,ax = plt.subplots(1,1,figsize=(4,4))
        
        for ii in range(len(self.Ez_list_Vcm)):
            ax.plot(self.Ez_list_Vcm[ii]*np.ones_like(self.energiesAll[ii]),self.energiesAll[ii]*1e-6,'.',color='gray')
            
        for ii in range(len(self.sb.highlight)):
            ax.plot(self.Ez_list_Vcm,self.energies[:,ii]*1e-6,'v',color='C'+str(ii),label=repr(self.sb.highlight[ii][:1]))
            ax.plot(self.Ez_list_Vcm,self.quad_fit(self.Ez_list_Vcm,*self.starkFits[ii])*1e-6,'-',color='C'+str(ii))

        ax.set_xlabel('Electric Field (V/cm)')
        ax.set_ylabel(r'Energy ($h\cdot$MHz)')
        ax.legend()
        ax.set_ylim(1e-6*np.array(energy_range_Hz))
        ax.grid(axis='both')

        return fig,ax

    
class analysis_pair_interaction(analysis):
    """
    Class for analyzing pair interactions in Rydberg atoms.

    This class extends the `analysis` class to specifically handle interactions between pairs of Rydberg atoms.
    It can include options for different interaction types and compute Hamiltonians for specified multipole interactions.

    Attributes:
        s1 (State): The first state in the pair.
        s2 (State): The second state in the pair, defaults to the first state if not specified.
        include_opts (dict): Options to include in the interaction calculations.
        pb (pair_basis): The pair basis used for the calculations, can be passed to skip computation.
        multipoles (list): List of multipole interactions to consider in the Hamiltonian computation.
        opts (dict): Options for the interaction calculations, initialized with default values and updated with `include_opts`.
        energies (list): List of energy values computed.
        energiesAll (list): List of all energy values computed.
        overlaps (list): List of overlap values computed.
        indices (list): List of indices used in computations.
        evAll (list): List of all eigenvalues computed.
        en0 (float): The initial energy computed without interactions.
    """
    def __init__(self,s1,s2=None,include_opts={},pb=None,multipoles=[[1,1]]):
        """
        Initializes the analysis_pair_interaction class.

        This constructor initializes an analysis_pair_interaction object which is used to analyze interactions between pairs of Rydberg atoms. It extends the functionality of the `analysis` class.

        Args:
            s1 (State): The first state in the pair.
            s2 (State, optional): The second state in the pair. Defaults to the first state if not specified.
            include_opts (dict, optional): Options to include in the interaction calculations. Defaults to an empty dictionary.
            pb (pair_basis, optional): The pair basis used for the calculations. If not provided, a new pair_basis object will be created and filled based on the provided states and options.
            multipoles (list, optional): List of multipole interactions to consider in the Hamiltonian computation. Defaults to [[1,1]] which represents dipole-dipole interactions.

        Attributes:
            opts (dict): Options for the interaction calculations, initialized with default values and updated with `include_opts`.
            s1 (State): The first state in the pair.
            s2 (State): The second state in the pair.
            pb (pair_basis): The pair basis used for the calculations.
        """
        self.opts = {'dn': 2,'dl': 2,'dm': 1,'dipole_allowed': False}
        
        for k,v in include_opts.items():
            self.opts[k] = v
        
        self.s1 = s1
        
        if s2 is None:
            self.s2 = s1
        else:
            self.s2 = s2
            
        if pb is None:
        
            self.pb = pair_basis()
            self.pb.fill(pair(self.s1,self.s2),include_opts=self.opts)

            print("Basis size: " + str(len(self.pb.pairs)))

            self.pb.computeHamiltonians(multipoles=multipoles)
        else:
            # for convenience when developing, allow this to be passed in to skip computation
            self.pb = pb

        
    def intfn(self,r,c6,c3):
        return c6/r**6 + c3/r**3

    def c3fn(self,r,C3k,det,Sk):
        return - det / (2*np.pi)*(1-np.sqrt(1+((4*C3k**2*Sk)/((det/(2*np.pi))**2*r**6))))
        
    def run(self,Bz_Gauss = 0, Ez_Vcm = 0, th=np.pi/2, phi=0, rList_um = np.arange(5,10,1),skip_fits = False,silence = False, diamagnetism = False):
        """
        Runs the analysis for the Rydberg atom interactions under specified conditions: distance, interatomic angle, E and B fields.

        This method computes the total Hamiltonian for different distances in `rList_um`, evaluates energy shifts and overlaps, and optionally fits interaction parameters.

        Args:
            Bz_Gauss (float): Magnetic field strength in Gauss. Default: 0.
            Ez_Vcm (float): Electric field strength in Volts per centimeter. Default: 0.
            th (float): Polar angle in radians. Default: pi/2.
            phi (float): Azimuthal angle in radians. Default: 0.
            rList_um (numpy.ndarray): Array of distances in micrometers at which to compute interactions. Default: np.arange(5,10,1).
            skip_fits (bool): If True, skip the curve fitting process. Default: False.
            silence (bool): If True, suppress progress output. Default: False.

        Returns:
            [c6d, c6e, c3d, c3d] vector of fitted interaction coefficients for the desired pair states, in units of Hz*um^6 and Hz*um^3 as appropriate.
        """
        
        self.env = environment(Bz_Gauss = Bz_Gauss, Ez_Vcm = Ez_Vcm, diamagnetism = diamagnetism)

        self.rList_um = rList_um
        self.th = th
        self.phi = phi
    
        self.energies = []
        self.energiesAll = []
        self.overlaps = []
        self.indices = []
        
        self.evAll = []
    
        self.en0 = self.pb.computeHtot(self.env,0,th=self.th,phi=self.phi,interactions=False)[0,0]
    
        for rum in tqdm(self.rList_um, disable = silence):
            
            ret = self.pb.computeHtot(self.env,rum,th=self.th,phi=self.phi,interactions=True)
    
            self.energiesAll.append(self.pb.es - self.en0)
            self.evAll.append(self.pb.ev)
    
            self.energies.append(ret[:,0] - self.en0)
            self.overlaps.append(ret[:,1])
            self.indices.append(ret[:,2])
    
        self.energies = np.array(self.energies)
        self.overlaps = np.array(self.overlaps)
        self.energiesAll = np.array(self.energiesAll)
        self.evAll = np.array(self.evAll)
        
        if skip_fits:
            return
        
        self.interactionFits = []
        
        for ii in range(len(self.pb.highlight)):
            popt,pcov = sp.optimize.curve_fit(self.intfn,self.rList_um,np.real(self.energies[:,ii]),p0=(2e5,1e9))#,sigma = np.abs(np.real(self.energies[:,ii]))
            # fitting on a log scale does better when the r range is large, although it's not clear
            #popt,pcov = sp.optimize.curve_fit(lambda r,c6,c3: np.log(intfn(np.exp(r),c6,c3)),np.log(rumList[:]),np.log(np.real(energies[:,ii])),p0=(1e9,1e7))
            self.interactionFits.append(popt)
        
        if len(self.interactionFits) > 1:
            c6d = (self.interactionFits[0][0] + self.interactionFits[1][0])/2
            c6e = (self.interactionFits[0][0] - self.interactionFits[1][0])/2
            c3d = (self.interactionFits[0][1] + self.interactionFits[1][1])/2
            c3e = (self.interactionFits[0][1] - self.interactionFits[1][1])/2
        else:
            c6d = self.interactionFits[0][0]
            c6e = 0
            c3d = self.interactionFits[0][1]
            c3e = 0
                    
        return np.array([c6d,c6e,c3d,c3e])
    
    
    def pa_plot(self,include_plot_opts ={}):
        """ Plot the results of the pair interaction analysis, including energy shifts and overlaps with asymptotic pair state. """

        self.plot_opts = {"ov_norm": 'linear',"s":5,"lin_norm":[0,1],"log_norm":[0.1,1],"gamma":0.5,"show_overlap": False,'cb_loc':'right','special_colors':None,'highlight_idx':0}
        self.overlapsAll = []

        for k, v in include_plot_opts.items():
            self.plot_opts[k] = v


        if self.plot_opts["show_overlap"] == True:
            fig,axs = plt.subplots(1,3,figsize=(12,4),gridspec_kw={'wspace':0.3})
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'wspace': 0.3})


        if self.plot_opts["special_colors"] == None:
            # red, blue, orange, teal, magenta, cyan
            colorschemes = [['#DDDDDD', '#CC3311'], ['#DDDDDD', '#0077BB'], ['#DDDDDD', '#EE7733'], ['#DDDDDD', '#009988'], ['#DDDDDD', '#EE3377'], ['#DDDDDD', '#33BBEE']]
        else:
            colorschemes = self.plot_opts["special_colors"]
        
        for ii in range(len(self.pb.highlight)):
            posidx = np.argwhere(self.energies[:,ii] >=0)
            negidx = np.argwhere(self.energies[:,ii] < 0)
            axs[0].plot(self.rList_um[posidx],np.abs(self.energies[posidx,ii])*1e-6,'o',color=colorschemes[ii % len(colorschemes)][1],label=repr(self.pb.highlight[ii][:2]))
            axs[0].plot(self.rList_um[negidx],np.abs(self.energies[negidx,ii])*1e-6,'o',color=colorschemes[ii % len(colorschemes)][1])
            #axs[0].plot(self.rList_um,np.abs(self.intfn(self.rList_um,*self.interactionFits[ii]))*1e-6,'-',color='C'+str(ii))
    
        axs[0].set_xlabel(r'$R$ ($\mu$m)')
        axs[0].set_ylabel(r'Pair Energy ($h\cdot$MHz)')
        #axs[0].legend()
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        #axs[0].set_ylim([-10,10])
        axs[0].grid(axis='both')

        if self.plot_opts["show_overlap"] == True:
            for ii in range(len(self.pb.highlight)):
                axs[1].plot(self.rList_um,self.overlaps[:,ii],'-',label=repr(self.pb.highlight[ii][:1]))

            axs[1].set_xlabel(r'$R$ ($\mu$m)')
            axs[1].set_ylabel(r'Overlap')
            #plt.legend()
            axs[1].set_xscale('log')
            #plt.yscale('log')
            axs[1].set_ylim([-0.1,1.2])
            axs[1].grid(axis='both')
        
        # now we want to plot energies of pair states and highlight using overlap


        minE = 0
        maxE = 0
    


        cmap0 = matplotlib.colors.LinearSegmentedColormap.from_list('testCmap', colorschemes[self.plot_opts["highlight_idx"] % len(colorschemes)], N=256)

        # determine range of initial target state
        newMinE = min(self.energies[:,self.plot_opts["highlight_idx"]]*1e-6)
        newMaxE = max(self.energies[:,self.plot_opts["highlight_idx"]]*1e-6)

        minE = min(newMinE,minE)
        maxE = max(newMaxE,maxE)

        ov = []
        Rlist = []
        Elist = []

        for jj in range(len(self.rList_um)):
            # self.evAll[jj] is list of eigenvalues for this r
            # take overlap with pb.highlight[3] which is ket for this highlight state
            ov = np.append(ov, ([np.abs(np.sum(self.pb.highlight[self.plot_opts["highlight_idx"]][3] * self.evAll[jj, :, kk])) ** 2 for kk in range(self.pb.dim())]))
            Rlist = np.append(Rlist, (list(self.rList_um[jj] * np.ones_like(self.energiesAll[jj]))))
            Elist = np.append(Elist, list((self.energiesAll[jj] * 1e-6)))



        # get order of points by overlap to plot points with high overlap on top of points with low overlap
        order = np.argsort(ov)

        self.overlapsAll.append([ov])

        # normalization overlap:
        if self.plot_opts["ov_norm"] == 'log':
            norm = matplotlib.colors.LogNorm(vmin=self.plot_opts["log_norm"][0], vmax=self.plot_opts["log_norm"][1],clip=True)
        elif self.plot_opts["ov_norm"] == 'linear':
            norm = matplotlib.colors.Normalize(vmin=self.plot_opts["lin_norm"][0], vmax=self.plot_opts["lin_norm"][1])
        elif self.plot_opts["ov_norm"] == 'power':
            norm = matplotlib.colors.PowerNorm(gamma=self.plot_opts["gamma"])

        sc = axs[-1].scatter(Rlist[order], Elist[order], s=self.plot_opts["s"], c=ov[order], cmap=cmap0,norm = norm)
        axs[-1].clb = fig.colorbar(sc, label=r'Overlap', ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],location = self.plot_opts['cb_loc'])

        axs[-1].set_xlabel(r'$R$ ($\mu$m)')
        axs[-1].set_ylabel(r'Pair Energy ($h\cdot$MHz)')
        #plt.legend()
        #axs[2].set_xscale('log')
        #plt.yscale('log')
        axs[-1].set_ylim([1.2*minE,1.2*maxE])
        axs[-1].grid(axis='both')
        axs[-1].set_axisbelow(True)
        
        return fig,axs # to allow later figure modification
            
    
        
        


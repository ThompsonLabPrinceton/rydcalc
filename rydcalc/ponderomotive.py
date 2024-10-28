#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:36:07 2020

@author: jt11
"""

import numpy as np
import functools
import scipy.constants as cs

from .constants import *


class ponderomotive:
    # base class that defines f_kq and lays out necessary functions.
    # can pass in external complex amplitude function
    
    def __init__(self,intensity_fn):
        self.intensity_fn = intensity_fn
    
    def intensity(self,z,r):
        return self.intensity_fn(z,r)
    
    def __hash__(self):
        # this is necessary for lru_cache to work--it needs some way to represent "self" as an integer
        return id(self)
    
    @functools.lru_cache(maxsize=1024)
    def f_kq(self,k,q,r):
        # This calculates the f_kq, but for most 
        angularIntegral, error = scipy.integrate.quad(lambda theta: theta_SpHarm(k,q,theta)*self.intensity(r*np.cos(theta), r*np.sin(theta))*np.sin(theta), 0, np.pi)        
        return(2*np.pi*angularIntegral)
    

class Ponderomotive3DLattice(ponderomotive):
    
    def __init__(self,kx,ky,kz,lambda_nm=1000,dx=0,dy=0,dz=0):
        
        self.kx = kx
        self.ky = ky
        self.kz = kz
        
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        # scale factors allowing turning off certain directions
        self.sx = 0 if kx==0 else 1
        self.sy = 0 if ky==0 else 1
        self.sz = 0 if kz==0 else 1
        
        if dx ==0 and dy==0 and dz==0:
            self.krange = range(0,30,2)
        else:
            self.krange = range(0,30)
        
        self.lambda_nm = lambda_nm
        
    def unit_scale(self):
        # converts integral to units of energy. Scaling to Hz is done in compute_energies
        self.omega = 2*np.pi*cs.c/(self.lambda_nm * 1e-9)
        return -0.5*cs.e**2/(cs.electron_mass * self.omega**2 * cs.epsilon_0 * cs.c)
        
    def intensity_cart(self,x,y,z):

        #return np.cos(2*self.kx*(x-self.dx))*np.cos(2*self.ky*(y-self.dy))*np.cos(2*self.kz*(z-self.dz))
    
        return self.sx*np.cos(2*self.kx*(x-self.dx)) + self.sy*np.cos(2*self.ky*(y-self.dy)) + self.sz*np.cos(2*self.kz*(z-self.dz))
        
    def intensity_sph(self,r,th,phi):

        return self.intensity_cart(r*np.sin(th)*np.cos(phi), r*np.sin(th)*np.sin(phi), r*np.cos(th))

    def sph2cart(self,r,th,phi):
        # takes xyz, returns (r,th,phi)
        return (r*np.sin(th)*np.cos(phi), r*np.sin(th)*np.sin(phi), r*np.cos(th))
    
    def sph2cyl(self,r,z,phi):
        
        return (r*np.sin(th)/nm, r*np.cos(th)/nm, phi)
    
    def cart2sph(self,x,y,z):
        return (np.sqrt(x**2+y**2+z**2), np.arctan2(y,x), np.arctan2(np.sqrt(x**2+y**2)/z))
    
    def cart2cyl(self,x,y,z):
        return (np.sqrt(x**2+y**2), z, np.arctan2(y,x))
    
    def cyl2cart(self,r,z,phi):
        return (r*np.cos(phi),r*np.sin(phi),z)
        
    def f_kq(self,k,q,r):
        if type(r) == type(np.zeros(2)):
            return np.array([self.f_kq_single(k,q,rr) for rr in r])
        else:
            return self.f_kq_single(k,q,r)
        
    @functools.lru_cache(maxsize=1000000)
    def f_kq_single(self,k,q,r):
        
        if np.abs(q) > k:
            return 0
        
        nquad = 50
        
        intfn = lambda th,phi: self.intensity_sph(r,th,phi) * np.conjugate(scipy.special.sph_harm(q,k,phi,th))*np.sin(th)
        val,err = my_fixed_dblquad(intfn,0,2*np.pi,0,np.pi,nx=nquad,ny=nquad)
        
        return val*np.sqrt((2*k+1)/(4*np.pi))
    
    def get_me(self,s1,s2):
        
        q = s2.m - s1.m
        
        me = 0
        for k in self.krange:
            me += s1.get_multipole_me(s2,k=k,operator=lambda r,k: self.f_kq(k,q,r*a0))

        return self.unit_scale()*me
    
# note that these are not tested
class PonderomotiveLG(Ponderomotive3DLattice):
    
    def __init__(self,p,l,lambda_nm,w0_nm,dr_nm=0,dz_nm=0):
        
        self.l = l
        self.p = p
        self.lambda_nm = lambda_nm
        self.w0_nm = w0_nm
        
        self.zR_nm = np.pi*self.w0_nm**2/self.lambda_nm
        
        self.dr_nm = dr_nm
        self.dz_nm = dz_nm
        
        self.krange = range(10)
        
        self.laguerre = scipy.special.genlaguerre(self.p,np.abs(self.l))
        self.norm = np.sqrt(2*np.math.factorial(self.p)/(np.pi*np.math.factorial(self.p + np.abs(self.l))))
        
    def efield_cyl(self,r_nm,z_nm,phi):
        
        if self.dr_nm != 0 or self.dz_nm != 0:
            x_nm,y_nm,z_nm = self.cyl2cart(r_nm,z_nm,phi)
            r_nm,z_nm,phi = self.cart2cyl(x_nm - self.dr_nm,y_nm,z_nm - self.dz_nm)
            
        # putting this first forces datatype of ret to complex right away
        ret = np.exp(-1.j*self.l*phi) * np.exp(1.j*self.psi(z_nm))
        
        ret *= self.norm
        
        wz = self.w(z_nm)
        
        ret *= (self.w0_nm / wz) * (r_nm * np.sqrt(2) / wz)**np.abs(self.l) * np.exp(-r_nm**2/wz**2)
        ret *= self.laguerre(2*r_nm**2/wz**2)
        #ret *= np.exp(-1.j*r_nm**2/(2*self.R(z_nm)))
        #ret *= np.exp(-1.j*self.l*phi) * np.exp(1.j*self.psi(z_nm))
        
        return ret
    
    def intensity_sph(self,r,th,phi):
        # in meters
        
        nm = 1e-9
        
        return np.abs(self.efield_cyl(r*np.sin(th)/nm, r*np.cos(th)/nm, phi))**2
        
    def w(self,z_nm):
        
        return self.w0_nm * np.sqrt(1+(z_nm/self.zR_nm)**2)
    
    def R(self,z_nm):
        
        return z_nm * (1 + (self.zR_nm/z_nm)**2)
    
    def psi(self,z_nm):
        
        return np.arctan(z_nm/self.zR_nm)

class PonderomotiveLG2(PonderomotiveLG):
    
    def __init__(self,LG1,LG2):
        
        self.LG1 = LG1
        self.LG2 = LG2
        
        self.krange = range(10)
        
        self.lambda_nm = self.LG1.lambda_nm
    
    def intensity_sph(self,r,th,phi):
        
        nm = 1e-9
        cyl_coords = (r*np.sin(th)/nm, r*np.cos(th)/nm, phi)
        return self.LG1.efield_cyl(*cyl_coords)*np.conjugate(self.LG2.efield_cyl(*cyl_coords))
    
    
# Note, if we want to make this mergeable into scipy, need it to support integration
# over vector-valued functions, see https://github.com/scipy/scipy/pull/6885
@functools.lru_cache(maxsize=1024)
def roots_legendre_cached(n):
    return scipy.special.roots_legendre(n)

def my_fixed_dblquad(func, ax, bx, ay, by, args=(), nx=5, ny=5):
    # f(y,x)
    x_x, w_x = roots_legendre_cached(nx)
    x_y, w_y = roots_legendre_cached(ny)
    
    x_x = np.real(x_x)
    x_y = np.real(x_y)
    
    if np.isinf(ax) or np.isinf(bx) or np.isinf(ay) or np.isinf(by):
        raise ValueError("Gaussian quadrature is only available for "
                         "finite limits.")
        
    x_x_scaled = (bx-ax)*(x_x+1)/2.0 + ax
    x_y_scaled = (by-ay)*(x_y+1)/2.0 + ay
    
    xpts,ypts = np.meshgrid(x_x_scaled,x_y_scaled)
    wts = np.outer(w_y,w_x)
    
    #return xpts,ypts,wts
    
    return (bx-ax)/2.0 * (by-ay)/2.0 * np.sum(wts*func(ypts,xpts, *args)), None
    #return w_x*w_y*func(ypts,xpts, *args)


class transition:
    
    def __init__(self,lam_nm,lifetime_ns,ji,jf,name=''):
        
        self.lam_nm = lam_nm
        self.lifetime_ns = lifetime_ns
        self.name = name
        
        # lower state j
        self.ji = ji
        
        # upper state j
        self.jf = jf
        
        self.omega_au = 2*np.pi/(lam_nm*1e-9*cs.alpha/a0)
        self.omega_sec = 2*np.pi*cs.c/(lam_nm*1e-9)
        
    def __repr__(self):
        
        return "%s: %.1f nm, tau=%.2f" % (self.name, self.lam_nm, self.lifetime_ns)
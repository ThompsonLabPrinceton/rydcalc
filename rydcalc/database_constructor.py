#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:19:13 2020

@author: jt11

This generates a bunch of pb objects and uses them to fill in the database for a new atom

As it is, it fills in the things needed to calculate C3,C6 interactions with low-l states.

In the future, could be expanded to calculate low-n to high-n matrix elements that would also
be needed for lifetime calculations...

"""

from rydcalc import *

Yb = Ytterbium174(cpp_numerov = True)

nlist = range(30,85)

for n0 in nlist:

    for s in [0,1]:
        s1 = state((n0,s,0,s,0),Yb,'fine')
        
        pb = pair_basis()
        pb.fill(pair(s1,s1),dn=3,dipole_allowed=True)
        
        print("n0=",n0,"pb dim:",pb.dim())
        
        pb.computeHamiltonians()

Yb._db_save_to_file()


Sr = Strontium88(cpp_numerov = True)

nlist = range(30,85)

for n0 in nlist:

    for s in [0,1]:
        s1 = state((n0,s,0,s,0),Sr,'fine')
        
        pb = pair_basis()
        pb.fill(pair(s1,s1),dn=3,dipole_allowed=True)
        
        print("n0=",n0,"pb dim:",pb.dim())
        
        pb.computeHamiltonians()

Sr._db_save_to_file()
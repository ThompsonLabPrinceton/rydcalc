Unnamed Rydberg Calculator

Jeff Thompson, 2/20-

This borrows concepts from ARC, and in particular incorporates the Numerov
wavefunction routines and associated parameterization of core potentials, etc.,
exactly.

The key difference, and motivation for starting over with a new codebase, is
to develop a framework that treats states as generic objects without regard
to the term symbols used to specify them (ie, Hydrogen nlm, fine structure,
hyperfine, etc.). The goal is to have a considerably more streamlined handling
of Alkaline earth atoms (with virtually zero code changes from Alkali), as well
as to be able to handle inter-species interactions and eventually more complex
phenomenon like singlet-triplet mixing and hyperfine structure in alkaline earths.


A few notes:

1. To compile ARC C numerov integrator, run this from terminal (os x):
'python setupc.py build_ext --inplace'

[Note: if you are using Anaconda with multiple environments, you must run this
with the correct environment activated on the command line!]

[Note: see ARC documentation for other platforms]

2. When initializing an atom (ie, Yb = rydcalc.Ytterbium174()), there are two
important options:
- cpp_numerov = True/False whether to use cpp numerov (much faster than python implementation)
- use_db = True/False whether to cache radial matrix elements (this will make calculations 
much faster over time, but it is _not_ a good idea if you are redefining/tinkering with
quantum defects)


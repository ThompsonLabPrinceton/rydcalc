Rydberg calculator

Thompson Lab
Contributors: Jeff Thompson, Michael Peper, Sam Cohen

This package computes quantities related to Rydberg states in an MDQT framework, including state energies, Stark shifts and pair interaction potentials. This software was used to implement the calculations in [M. Peper et al PRX 15, 011009 (2025)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.011009) for <sup>174</sup>Yb and <sup>171</sup>Yb. An earlier version was used to implement the calculations in [Cohen and Thompson, PRX 2 030322 (2022)](https://link.aps.org/doi/10.1103/PRXQuantum.2.030322).

### Update 07/2025

Support for Yb Rydberg states with $\ell = 3$ ($F$) and $\ell = 4$ ($G$) has been added [R. Kuroda, V.M. Hughes et al arXiv:2507.11487](https://arxiv.org/abs/2507.11487).

The basic approach to computing pair potentials follows [pairinteraction](https://github.com/pairinteraction/pairinteraction) and the [Alkali Rydberg Calculator (ARC)](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator). In particular, we have directly incorporated functions for computing wavefunctions numerically using the Numerov method from ARC.

`rydcalc` also has the necessary atomic data to perform calculations for Alkali atoms, which is intended mainly for debugging and comparison to results obtained with other programs that have been more extensively tested and documented. If you can do your calculation with other programs, we recommend that! But if you need MQDT, please try `rydcalc`.

Documentation illustrating the basic functionality of the code can be found in the tutorial.ipynb notebook, and in the comments throughout the code.

Note: to compile the ARC C numerov integrator, run this from a console (in MacOS/Linux):
'python setupc.py build_ext --inplace'

[Note: if you are using Anaconda with multiple environments, you must run this
with the correct environment activated on the command line!]

See [ARC documentation](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/installation.html#compiling-c-extension) for other platforms.

This work was supported by the Army Research Office (W911NF-24-10358), the Office of Naval Research (N00014-20-1-2426), the National Science Foundation (QLCI grant OMA-2120757, and NSF CAREER PHY-2047620), the Sloan Foundation, the Gordon and Betty Moore Foundation (Grant DOI 10.37807/gbmf12253). 
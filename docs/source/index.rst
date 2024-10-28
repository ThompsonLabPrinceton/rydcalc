.. rydcalc documentation master file, created by
   sphinx-quickstart on Mon May 22 13:58:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Rydcalc
===================================

This is the documentation for rydcalc, an MQDT-based framework for calculating Stark shifts and interactions between Rydberg atoms developed by the Thompson Lab at Princeton University. While the code was initially developed for Yb, with a particular emphasis on 171Yb, it can be easily adapted to any other atomic species, or to calculate inter-species interactions. A version of this code was used to calculate the interactions of circular Rydberg states with low-l states in [Cohen and Thompson, 2021].

The development of this code was heavily influenced by existing Rydberg interaction calculation software, including the `Alkali Rydberg Calculator (ARC) <https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator>`__ and `pairinteraction <https://github.com/pairinteraction/pairinteraction/>`__. The Numerov integrator and model potential code from ARC are incorporated into rydcalc.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

.. toctree::
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

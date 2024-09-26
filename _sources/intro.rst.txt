.. sectnum::

Welcome to the Pareto DAB (Dual Active Bridge) Tool
===================================================


Installation
---------------------------------------
Install the Toolbox as a developer

::

    pip install -e .

Pareto DAB (Dual Active Bridge) Tool function documentation
===========================================================
.. currentmodule:: paretodab.debug_tools

.. automodule:: paretodab.debug_tools
   :members:

Store specifications, modulation and simulation results
--------------------------------------------------------
.. autoclass:: paretodab.DabData
   :members:

Calculate DAB intervals
------------------------------
.. currentmodule:: paretodab.interval_calc

.. automodule:: paretodab.interval_calc
   :members: calc_modulation, _calc_interval_1, _calc_interval_2, _calc_interval_3, _integrate_Coss

Sample requirements
------------------------------
.. automodule:: paretodab.sample_requirements
   :members:
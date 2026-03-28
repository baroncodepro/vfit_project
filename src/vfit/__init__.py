"""
vfit — Vector Fitting for RLC and S-domain systems
====================================================

Quick start
-----------
>>> import numpy as np
>>> from vfit import VectorFitter, foster_synthesis
>>> from vfit.visualization import bode_plot, pole_zero_map
>>>
>>> freq = np.logspace(7, 11, 300)
>>> s    = 1j * 2 * np.pi * freq
>>> Z    = 50 + s * 10e-9 + 1 / (s * 1e-12)   # series RLC
>>>
>>> model = VectorFitter(n_poles=6).fit(freq, Z)
>>> print(model)
>>> bode_plot(freq, Z, model)
>>> network = foster_synthesis(model)
>>> print(network)
"""

from .core.vector_fitting   import VectorFitter, VFOptions
from .core.rational_function import RationalModel
from .core.pole_zero import (
    enforce_conjugate_pairs,
    stabilize_poles,
    pole_resonant_frequency,
    pole_quality_factor,
)
from .rlc.rlc_synthesis   import foster_synthesis, FosterNetwork, RLCBranch
from .export.spice_export import export_spice_foster, export_spice_behavioral
from .utils.data_loader   import load_csv, load_ri_csv, load_touchstone, MeasurementData
from .solvers.passivity   import check_passivity, enforce_passivity, PassivityReport, EnforcementResult
from .solvers.auto_order  import auto_order, OrderSweepResult

__all__ = [
    "VectorFitter", "VFOptions",
    "RationalModel",
    "enforce_conjugate_pairs", "stabilize_poles",
    "pole_resonant_frequency", "pole_quality_factor",
    "foster_synthesis", "FosterNetwork", "RLCBranch",
    "export_spice_foster", "export_spice_behavioral",
    "load_csv", "load_ri_csv", "load_touchstone", "MeasurementData",
    "check_passivity", "enforce_passivity", "PassivityReport", "EnforcementResult",
    "auto_order", "OrderSweepResult",
]

__version__ = "0.1.0"

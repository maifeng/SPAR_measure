"""SPAR: Semantic Projection with Active Retrieval.

Reference implementation for Yan, Mai, Wu, Chen & Li (2024),
"A Computational Framework for Understanding Firm Communication During
Disasters," Information Systems Research 35(2): 590-608,
https://doi.org/10.1287/isre.2022.0128.
"""

from __future__ import annotations

__version__ = "0.2.1"
__paper__ = "Yan, Mai, Wu, Chen & Li (2024), ISR 35(2):590-608"

from spar_measure.gui import Measurement, run_gui

# Backwards-compat for the original misspelled class name.
Meaurement = Measurement

__all__ = ["Measurement", "Meaurement", "run_gui", "__version__", "__paper__"]

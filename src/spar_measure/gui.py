# SPDX-License-Identifier: GPL-3.0-or-later
"""Backwards-compat shim re-exporting the Gradio UI from :mod:`spar_measure.ui`.

Prior to 0.3.0 the entire package lived in a 1460-line ``gui.py``. The
refactor split that file into :mod:`ui`, :mod:`core`, :mod:`state`,
:mod:`io`, :mod:`api`, and :mod:`cli`. This module keeps the old import
paths working:

- ``from spar_measure.gui import run_gui``
- ``from spar_measure.gui import Measurement``
- ``python -m spar_measure.gui`` (still launches the UI)
"""

from __future__ import annotations

import fire

from .ui import CVFDemo, Measurement, PathManager, build_blocks, run_gui

# Backwards-compat alias for the original misspelled class name.
Meaurement = Measurement

__all__ = [
    "CVFDemo",
    "Meaurement",
    "Measurement",
    "PathManager",
    "build_blocks",
    "run_gui",
]


if __name__ == "__main__":
    # Use the new CLI dispatcher so ``python -m spar_measure.gui --help`` works.
    from .cli import gui as _gui

    fire.Fire(_gui)

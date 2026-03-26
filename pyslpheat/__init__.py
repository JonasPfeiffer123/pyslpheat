"""
pyslpheat
=========
Standard load profiles (SLP) for heat demand in district heating simulations.

Implements two German standards:

bdew
    BDEW Standard Load Profile methodology (hourly, climate-dependent).
    Supports annual-energy scaling (Mode A), design-load scaling (Mode B),
    and combined scaling (Mode C). Optional stochastic post-processing
    (peak jitter + log-normal noise) via ``stochastic=True``.

vdi4655
    VDI 4655 methodology (15-min, day-type based).

Both ``calculate()`` functions return a ``pandas.DataFrame`` with a
``DatetimeIndex`` and columns ``Q_heat_kWh``, ``Q_dhw_kWh``,
``Q_total_kWh``, ``temperature_C``.
VDI 4655 additionally includes ``Q_electricity_kWh``.
"""

import os as _os

from .bdew    import calculate as bdew_calculate      # noqa: F401
from .vdi4655 import calculate as vdi4655_calculate   # noqa: F401

__version__ = "0.2.0"
__author__  = "Jonas Pfeiffer"

__all__ = [
    "bdew_calculate",
    "vdi4655_calculate",
    "TRY_BAUTZEN_2015",
    "TRY_BAUTZEN_2015_WINTER",
    "TRY_BAUTZEN_2015_SUMMER",
    "TRY_BAUTZEN_2045",
    "TRY_BAUTZEN_2045_WINTER",
    "TRY_BAUTZEN_2045_SUMMER",
]

# Bundled TRY weather files (DWD station Bautzen, 51.1676°N 14.4222°E, climate zone 9)
_TRY_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "try")

TRY_BAUTZEN_2015 = _os.path.join(_TRY_DIR, "TRY2015_511676144222_Jahr.dat")
"""DWD TRY 2015 – annual (average year) – Bautzen (51.1676°N, 14.4222°E, climate zone 9)."""

TRY_BAUTZEN_2015_WINTER = _os.path.join(_TRY_DIR, "TRY2015_511676144222_Wint.dat")
"""DWD TRY 2015 – extreme cold winter – Bautzen."""

TRY_BAUTZEN_2015_SUMMER = _os.path.join(_TRY_DIR, "TRY2015_511676144222_Somm.dat")
"""DWD TRY 2015 – extreme hot summer – Bautzen."""

TRY_BAUTZEN_2045 = _os.path.join(_TRY_DIR, "TRY2045_511676144222_Jahr.dat")
"""DWD TRY 2045 – annual (average year, future climate scenario) – Bautzen."""

TRY_BAUTZEN_2045_WINTER = _os.path.join(_TRY_DIR, "TRY2045_511676144222_Wint.dat")
"""DWD TRY 2045 – extreme cold winter (future climate scenario) – Bautzen."""

TRY_BAUTZEN_2045_SUMMER = _os.path.join(_TRY_DIR, "TRY2045_511676144222_Somm.dat")
"""DWD TRY 2045 – extreme hot summer (future climate scenario) – Bautzen."""

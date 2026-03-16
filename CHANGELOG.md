# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2026-03-16

### Added
- BDEW SigLinDe heat demand module (`bdew.py`) with full coefficient table
  (residential HEF/HMF and all commercial G-types, subtypes 03–05, 33/34)
- VDI 4655 day-type demand module (`vdi4655.py`) with 15-min quarter-hourly
  profiles for EFH/MFH, climate zones 1–15
- Three scaling modes: Mode A (annual energy), Mode B (design load),
  Mode C (both via β-bisection)
- Configurable options: `heating_limit_temp`, `heating_exponent`,
  `dhw_share`, `dhw_flat`
- Stochastic post-processing: peak jitter (circular daily shift) +
  log-normal amplitude noise with energy renormalization
- Six bundled DWD TRY files for Bautzen (51.1676°N, 14.4222°E, climate
  zone 9): average year, extreme winter, extreme summer × 2015 and 2045
- Comprehensive example scripts with parameter comparison figures
  (`examples/bdew_demo.py`, `examples/vdi4655_demo.py`)

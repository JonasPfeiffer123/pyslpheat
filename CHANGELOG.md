# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] — 2026-03-26

### Added
- Discrete DHW draw events (`dhw_draw_events=True`) in `bdew.calculate()`:
  replaces the smooth BDEW DHW baseline with stochastic clustered draw events
  (bimodal morning/evening, Poisson draw count per day, log-normal amplitude);
  annual DHW energy preserved by renormalisation
- New parameters: `dhw_draws_per_day` (float, default 4.0) and `dhw_draw_seed`
  (int, default 42), independent of the existing `stochastic_seed`
- BDEW GUI tab: new "Diskrete TWW-Zapfereignisse" group with checkbox and
  sub-parameters (same enable/disable pattern as Stochastik group)
- Documentation: new section *Discrete DHW draw events* in `docs/DOCUMENTATION.md`

## [0.2.0] — 2026-03-26

### Added
- PyQt6 desktop GUI (`pyslpheat-gui`) covering the full parameter set of both
  modules — no scripting required
- `[gui]` optional dependency group: `PyQt6>=6.5`, `matplotlib>=3.7`
- `pyslpheat-gui` console script entry point registered via `pyproject.toml`
- Automatic import of statutory German public holidays via `compute_holidays`
  in the VDI 4655 tab
- Embedded matplotlib plot with navigation toolbar; CSV export for all results

### Fixed
- `building_type = "B"` (office) was missing from the VDI 4655 parameter
  table in `docs/DOCUMENTATION.md`

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

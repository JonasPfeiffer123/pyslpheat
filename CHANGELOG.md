# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] — 2026-09-30

### Changed
- `bdew.calculate()` and `vdi4655.calculate()` are much faster when called
  repeatedly, e.g. once per building. Per call after the first one in a
  process: BDEW ≈ 28 ms → ≈ 1.6 ms, VDI 4655 ≈ 115 ms → ≈ 2.4 ms. The first
  call takes ≈ 27 ms (BDEW) and ≈ 35 ms (VDI 4655, previously ≈ 115 ms).
  Signatures and results are unchanged
- TRY weather files are parsed once and cached per process; a file is read
  again when its modification time or size changes
- Bundled coefficient, factor and profile tables are read once per process and
  addressed through lookup tables instead of pandas merges and row filters
- Per-day and per-interval Python loops are replaced by numpy operations
- Stochastic post-processing is faster: `stochastic=True` ≈ 185 ms → ≈ 4.5 ms,
  `dhw_draw_events=True` ≈ 45 ms → ≈ 9 ms; results for a given seed are unchanged

### Added
- `pyslpheat.clear_caches()` drops all cached weather files and tables
- Test suite (`pytest`): golden-master tests for both `calculate()` functions
  recorded from 0.3.0, comparisons of the helper functions with their 0.3.0
  implementations, and tests of the cache behaviour
- `benchmarks/bench_calculate.py` for wall-clock measurements
- Documentation: new section *Performance and caching* in
  `docs/DOCUMENTATION.md` (memory use, invalidation, thread safety)
- README: section *Known issues* (leap years with 8760-hour weather files,
  profile type `GMF`, `peak_design_kW` without `design_temperature`, different
  time steps of the two modules)

### Fixed
- README: the demo scripts take the weather file as `--try-file`, not as a
  positional argument

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

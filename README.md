# pyslpheat

**Standard Load Profiles for Heat demand** — Python implementation of two German
heat demand profile standards for district heating simulations.

| Module | Standard | Resolution | Method |
|--------|----------|------------|--------|
| `bdew` | BDEW/VKU/GEODE SigLinDe | Hourly | Temperature-continuous sigmoid + linear |
| `vdi4655` | VDI 4655 | 15-minute | Day-type step profiles |

## Installation

```bash
pip install .
# or editable install during development:
pip install -e .
```

## Quick start

```python
from pyslpheat import bdew_calculate, TRY_BAUTZEN_2015

# BDEW – hourly profile, 20 000 kWh/a, multi-family residential
# Uses the bundled Bautzen TRY (51.1676°N 14.4222°E, climate zone 9)
df = bdew_calculate(
    annual_heat_kWh = 20_000,
    profile_type    = "HMF",
    subtype         = "03",
    TRY_file_path   = TRY_BAUTZEN_2015,
    year            = 2026,
)
print(df["Q_total_kWh"].sum())   # → ~20 000 kWh
print(df["Q_total_kWh"].max())   # → peak load [kWh/h]
```

Both functions return a `pandas.DataFrame` with a `DatetimeIndex` and columns
`Q_heat_kWh`, `Q_dhw_kWh`, `Q_total_kWh`, `temperature_C`
(VDI 4655 additionally includes `Q_electricity_kWh`).

## GUI – no scripting required

pyslpheat ships a desktop application that lets you configure all parameters
interactively, inspect the resulting load profile in an embedded plot, and
export results as CSV — without writing a single line of code.

**Install with GUI dependencies:**

```bash
pip install ".[gui]"
```

**Launch:**

```bash
pyslpheat-gui
```

The application has two tabs — one for each calculation standard:

| Tab | Standard | Output resolution |
|-----|----------|-------------------|
| BDEW SigLinDe | BDEW/VKU/GEODE SigLinDe | Hourly (8 760 values) |
| VDI 4655 | VDI 4655 | 15-minute (35 040 values) |

Both tabs offer the full parameter set including stochastic post-processing
(BDEW) and automatic import of statutory German public holidays (VDI 4655).
The six bundled Bautzen TRY files are selectable from a drop-down; custom
TRY files can be loaded via a file browser.

**BDEW SigLinDe tab**

![BDEW SigLinDe tab](docs/img/pyslpheat_gui_BDEW.jpg)

**VDI 4655 tab**

![VDI 4655 tab](docs/img/pyslpheat_gui_VDI4655.jpg)

## Demo scripts

```bash
python examples/bdew_demo.py    path/to/TRY.dat --output ./output
python examples/vdi4655_demo.py path/to/TRY.dat --output ./output
```

The package ships with six TRY files for **Bautzen** (51.1676°N, 14.4222°E,
DWD climate zone 9) in `data/try/`: average year, extreme cold winter, and extreme
hot summer for both the 2015 and 2045 climate epochs.

```python
from pyslpheat import TRY_BAUTZEN_2015, TRY_BAUTZEN_2015_WINTER, TRY_BAUTZEN_2015_SUMMER
from pyslpheat import TRY_BAUTZEN_2045, TRY_BAUTZEN_2045_WINTER, TRY_BAUTZEN_2045_SUMMER
```

Additional TRY files are available from
[DWD / BBSR](https://www.bbsr.bund.de/BBSR/DE/forschung/programme/zb/Auftragsforschung/5EnergieKlimaBauen/2013/testreferenzjahre/01-start.html).

## Documentation

Full API reference: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)

## Related projects & ecosystem

Several Python packages implement BDEW or VDI 4655 heat load profiles.
The table below covers the actively maintained, pip/conda-installable tools.

| Package | Install | Standards | Subtypes | Hourly shape | Peak scaling | Heating limit | Stochastic | TRY bundled |
|---|---|---|---|---|---|---|---|---|
| **pyslpheat** (this) | `pip install .` | BDEW SigLinDe, VDI 4655 | HEF/HMF 03–05, 33/34; all G-types 01–05, 33/34 | BGW hourly factors | ✓ Mode B/C (β-bisection) | ✓ | ✓ peak jitter + log-normal | ✓ 6 files (2015/2045 ×3) |
| **[oemof-demand](https://github.com/oemof/demandlib)** | `pip install oemof-demand` | BDEW SLP, VDI 4655 | building\_class 1–11, wind\_class 0/1 | BDEW hourly factors | ✗ | ✗ | ✗ | ✗ |
| **[when2heat](https://github.com/oruhnau/when2heat)** | pip from repo | BDEW SLP | SFH / MFH / COM only | BGW hourly factors | ✗ | ✗ | ✗ | ✗ |
| **[lpagg](https://github.com/jnettels/lpagg)** | conda | VDI 4655 | EFH / MFH | 15-min day-type profiles | ✗ | ✗ | ✓ random time-shift | ✗ |
| **[DistrictHeatingSim](https://github.com/JonasPfeiffer123/DistrictHeatingSim)** | pip from repo | BDEW SigLinDe, VDI 4655 | HEF/HMF 03–05, 33/34; all G-types | BGW hourly factors | ✗ | ✗ | ✗ | ✗ |

### Broader ecosystem

Tools beyond pip/conda packages — physics-based simulators, activity-based
generators, and external web/desktop tools that produce comparable heat demand
time series:

| Tool | Type | Method | Notes |
|---|---|---|---|
| **[HiSim](https://github.com/FZJ-IEK3-VSA/HiSim)** | Python package | Physics-based thermal simulation (TABULA building database) | Full dynamic simulation incl. solar gains, internal loads, thermal mass; most detailed approach |
| **[pylpg](https://github.com/FZJ-IEK3-VSA/pylpg)** | Python package | Activity-based (LoadProfileGenerator, TU Chemnitz) | Stochastic activity chains per person → energy flows; minute-level resolution |
| **[EnSySim](https://github.com/HSZittauGoerlitz/EnSySim)** | Python package (HS Zittau/Görlitz) | Heating degree-hours (HGT) | Simple linear approach; no BDEW coefficients; useful as lower-complexity baseline |
| **[nPro](https://www.npro.energy/)** | Web tool (RWTH Aachen) | Proprietary (VDI 4655 & BDEW referenced) | Pre-calculated profiles by building age class; export as CSV |
| **[Sophena](https://www.carmen-ev.de/service/sophena/)** | Desktop software (IER Stuttgart) | BDEW methodology + regional statistics | District heat planning tool; export profiles as CSV |
| **[synPRO](https://synpro-lastprofile.de/)** | Web tool (Fraunhofer ISE) | Statistical-synthetic 15-min profiles | Stochastically generated; export as `.dat`; residential old stock / passive house |
| **[FreePlan](https://tu-dresden.de/ing/maschinenwesen/iet/gewv/forschung/forschungsprojekte/delfin)** | Excel tool (TU Dresden) | Regression model (model public, training data/coefficients not) | Spreadsheet-based profile generation; no scripting required |

### What pyslpheat does differently

**Original coefficients, then options on top.**
pyslpheat reads the unmodified BDEW coefficient table (`daily_coefficients.csv`)
covering all residential and commercial profile types with all subtype variants.
Configuration options are layered on top — none of them alter the underlying
BDEW shape unless explicitly requested:

| Option | What it does |
|---|---|
| `subtype` | Selects the insulation/consumption class: `"03"` high (old stock), `"04"` medium, `"05"` low (new build); `"33"`/`"34"` are the SigLinDe variants with non-zero linear correction terms |
| `dhw_share` | Fixes the DHW fraction of the total annual demand, overriding the profile's natural split |
| `dhw_flat` | Distributes DHW uniformly across 24 h (temperature- and weekday-independent) |
| `heating_limit_temp` | Sets a Heizgrenztemperatur — space heating is zeroed on warm days; total energy is renormalized automatically |
| `heating_exponent` | Power-law reshaping of the daily heating distribution: `>1` → sharper winter peaks, `<1` → flatter (same annual total) |
| `peak_design_kW` + `design_temperature` | **Mode B/C** — constrains the design-day peak via bisection on a shape exponent β, simultaneously satisfying both the annual energy and the design load target |
| `stochastic=True` | Adds day-level peak-time jitter (circular hourly shift) and log-normal amplitude noise with energy renormalization, producing statistically distinct realisations |

**Relation to DistrictHeatingSim.**
The BDEW and VDI 4655 modules were originally developed as part of
[DistrictHeatingSim](https://github.com/JonasPfeiffer123/DistrictHeatingSim),
an open-source district heating simulation framework (own development).
After its publication, the heat profile logic was extracted into this
standalone package and further extended — most notably with the stochastic
post-processing (peak jitter + log-normal amplitude noise) which was not
present in DistrictHeatingSim. pyslpheat can therefore be used independently
of the full simulation framework.

**demandlib comparison.**
`demandlib.bdew.HeatBuilding` parametrises temperature sensitivity via
`building_class` (1–11) and `wind_class` (0/1) — continuous knobs that shift
the sigmoid curve. pyslpheat instead uses the discrete BDEW subtype codes
(`03`/`04`/`05`/`33`/`34`) that correspond directly to the tabulated coefficient
sets in the official guideline, then adds independent post-processing options.
Neither approach is strictly superior; they answer different questions.

> **Know another tool that belongs here?**
> Feel free to open an issue or pull request to add further heat load profile
> calculation tools to the comparison table.

## Development context

pyslpheat emerged from research in heat supply at
**Hochschule Zittau/Görlitz**. The heat profile modules were originally
developed as part of
[DistrictHeatingSim](https://github.com/JonasPfeiffer123/DistrictHeatingSim)
within the **SMWK-NEUES TG70** project (*development and testing of methods
and tools for the conceptualization of sustainable heating networks*).
They were subsequently extracted into this standalone package and further
extended — most notably with stochastic post-processing — as part of an
**EFRE/ESF-funded Junior Research Group "Energy Storage"** at
Hochschule Zittau/Görlitz.

### Funding

![Funding notice — Saxony / EU](pyslpheat/images/funding_saxony_EU.jpg)

## License

MIT — see [LICENSE](LICENSE).

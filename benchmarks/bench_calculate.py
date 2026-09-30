"""
Wall-clock benchmark for ``bdew.calculate`` and ``vdi4655.calculate``.

Calls both functions the way DistrictHeatingSim does (one call per building,
same weather file for all buildings) and reports, without a profiler:

* first call  – first call in a fresh interpreter (cold caches), import time
  excluded; median over several fresh processes
* later calls – median and mean per call within one process (warm caches)

Usage::

    python benchmarks/bench_calculate.py [--calls 200] [--processes 7] [--json out.json]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import pyslpheat  # noqa: E402
from pyslpheat import bdew_calculate, vdi4655_calculate  # noqa: E402

YEAR = 2023
TRY = pyslpheat.TRY_BAUTZEN_2015
HOLIDAYS = np.array(["2023-01-01", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-18",
                     "2023-05-29", "2023-10-03", "2023-12-25", "2023-12-26"],
                    dtype="datetime64[D]")

BDEW_BUILDINGS = [("GBD", "03"), ("GKO", "03"), ("HMF", "03"), ("GHA", "04"), ("HEF", "33")]
VDI_BUILDINGS = ["EFH", "MFH"]


def bdew_call(i: int):
    profile_type, subtype = BDEW_BUILDINGS[i % len(BDEW_BUILDINGS)]
    return bdew_calculate(
        annual_heat_kWh=40_000.0 + 1_000.0 * (i % 50), profile_type=profile_type,
        subtype=subtype, TRY_file_path=TRY, year=YEAR, dhw_share=0.2,
        heating_limit_temp=15.0, heating_exponent=1.0, peak_design_kW=None)


def vdi_call(i: int):
    return vdi4655_calculate(
        annual_heating_kWh=12_000.0 + 500.0 * (i % 50), annual_dhw_kWh=3_000.0,
        annual_electricity_kWh=1, building_type=VDI_BUILDINGS[i % len(VDI_BUILDINGS)],
        number_people_household=2, year=YEAR, climate_zone="9", TRY=TRY, holidays=HOLIDAYS)


CALLS = {"bdew": bdew_call, "vdi4655": vdi_call}


def _timed_ms(func, i: int) -> float:
    start = time.perf_counter()
    func(i)
    return (time.perf_counter() - start) * 1000.0


def first_call_ms(name: str) -> float:
    """Time the first call in this interpreter (run in a fresh process)."""
    return _timed_ms(CALLS[name], 0)


def later_calls_ms(name: str, calls: int) -> list:
    func = CALLS[name]
    for i in range(3):
        func(i)
    return [_timed_ms(func, i) for i in range(calls)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--calls", type=int, default=200, help="warm calls per function")
    parser.add_argument("--processes", type=int, default=7, help="fresh processes for first call")
    parser.add_argument("--json", help="write results to this file")
    parser.add_argument("--first-call", choices=sorted(CALLS), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.first_call:
        print(first_call_ms(args.first_call))
        return

    results = {}
    for name in CALLS:
        first = [
            float(subprocess.check_output(
                [sys.executable, os.path.abspath(__file__), "--first-call", name], text=True))
            for _ in range(args.processes)
        ]
        later = later_calls_ms(name, args.calls)
        results[name] = {
            "first_call_ms_median": statistics.median(first),
            "later_call_ms_median": statistics.median(later),
            "later_call_ms_mean": statistics.fmean(later),
            "later_call_ms_min": min(later),
        }
        print(f"{name:8s} first call {results[name]['first_call_ms_median']:8.2f} ms | "
              f"later calls median {results[name]['later_call_ms_median']:7.3f} ms, "
              f"mean {results[name]['later_call_ms_mean']:7.3f} ms, "
              f"min {results[name]['later_call_ms_min']:7.3f} ms  "
              f"({args.calls} calls, {args.processes} processes)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()

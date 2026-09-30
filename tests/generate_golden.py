"""
Record the golden master for ``tests/test_golden_master.py``.

Run this only against a reference implementation whose outputs are known to
be correct::

    python tests/generate_golden.py

The committed golden master was recorded from the unoptimised implementation
at commit 2751480 (v0.3.0). Regenerating it from optimised code defeats its
purpose; do so only when a change of results is intended.

To keep the file small, the regular DatetimeIndex is stored as
start/step/periods, identical arrays are stored once, and ``Q_total_kWh`` is
stored as "sum of heat and DHW" where that reproduces the recorded values bit
for bit. Each of these shortcuts is verified here before it is used.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_cases import (  # noqa: E402
    CASES, GOLDEN_DIR, GOLDEN_META, GOLDEN_NPZ, expected_index, run_case, write_leap_try,
)

_SUM_COLUMNS = ["Q_heat_kWh", "Q_dhw_kWh"]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(("git",) + args, text=True,
                                       cwd=os.path.dirname(GOLDEN_DIR)).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _index_spec(index: pd.DatetimeIndex) -> dict:
    values = index.values
    step = (values[1] - values[0]) / np.timedelta64(1, "s")
    spec = {
        "dtype": str(values.dtype),
        "start": str(values[0].astype("datetime64[s]")),
        "step_seconds": int(step),
        "periods": len(values),
        "freq": index.freqstr,
    }
    rebuilt = expected_index(spec)
    if rebuilt.dtype != values.dtype or not np.array_equal(rebuilt, values):
        raise AssertionError("index is not a regular range; store it explicitly")
    return spec


def main() -> None:
    arrays = {}
    by_digest = {}
    cases = {}

    def store(key: str, values: np.ndarray) -> str:
        digest = hashlib.sha256(str(values.dtype).encode() + values.tobytes()).hexdigest()
        if digest not in by_digest:
            by_digest[digest] = key
            arrays[key] = values
        return by_digest[digest]

    with tempfile.TemporaryDirectory() as tmp:
        leap_try_path = write_leap_try(tmp)
        for name in CASES:
            status, result = run_case(name, leap_try_path)
            if status == "error":
                cases[name] = {"status": "error", "type": type(result).__name__,
                               "message": str(result)}
                print(f"{name}: {type(result).__name__}: {result}")
                continue
            columns = {}
            total = sum(result[c].values for c in _SUM_COLUMNS)
            for column in result.columns:
                values = result[column].values
                if column == "Q_total_kWh" and values.tobytes() == total.tobytes():
                    columns[column] = {"sum_of": _SUM_COLUMNS}
                else:
                    columns[column] = {"key": store(f"{name}::{column}", values)}
            cases[name] = {"status": "ok", "index": _index_spec(result.index),
                           "columns": columns}
            print(f"{name}: ok {result.shape}")

    meta = {
        "recorded_with": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--", "pyslpheat")),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": sys.platform,
            "machine": platform.machine(),
        },
        "cases": cases,
    }
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    np.savez_compressed(GOLDEN_NPZ, **arrays)
    with open(GOLDEN_META, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {GOLDEN_NPZ} ({os.path.getsize(GOLDEN_NPZ) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

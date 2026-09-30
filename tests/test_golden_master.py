"""
Golden-master tests for ``bdew.calculate`` and ``vdi4655.calculate``.

The expected outputs were recorded from the unoptimised reference
implementation (see ``generate_golden.py``). Every case is checked twice:

* ``test_matches_golden`` – index exactly, all columns with ``rtol=1e-12``.
* ``test_bitwise_identical`` – all columns byte for byte. Bitwise results
  depend on the numpy build, so this runs only in the environment the golden
  master was recorded in (or with ``PYSLPHEAT_GOLDEN_BITWISE=1``).
"""

import json
import os
import platform
import sys

import numpy as np
import pytest

from golden_cases import (
    CASES, GOLDEN_META, GOLDEN_NPZ, expected_index, run_case, write_leap_try,
)

with open(GOLDEN_META, "r", encoding="utf-8") as _fh:
    META = json.load(_fh)

_RECORDED = META["recorded_with"]
_SAME_ENVIRONMENT = (
    _RECORDED["numpy"] == np.__version__
    and _RECORDED["platform"] == sys.platform
    and _RECORDED["machine"] == platform.machine()
)
_BITWISE = _SAME_ENVIRONMENT or os.environ.get("PYSLPHEAT_GOLDEN_BITWISE") == "1"
_OK_CASES = sorted(n for n, c in META["cases"].items() if c["status"] == "ok")


@pytest.fixture(scope="module")
def expected_columns():
    """Return a function mapping a case name to its recorded column arrays."""
    with np.load(GOLDEN_NPZ) as data:
        arrays = {key: data[key] for key in data.files}

    def get(name):
        specs = META["cases"][name]["columns"]
        columns = {c: arrays[s["key"]] for c, s in specs.items() if "key" in s}
        for column, spec in specs.items():
            if "sum_of" in spec:
                columns[column] = sum(columns[c] for c in spec["sum_of"])
        return {column: columns[column] for column in specs}

    return get


@pytest.fixture(scope="module")
def leap_try_path(tmp_path_factory):
    return write_leap_try(str(tmp_path_factory.mktemp("try")))


@pytest.fixture(scope="module")
def results(leap_try_path):
    """Run each case at most once per test module."""
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = run_case(name, leap_try_path)
        return cache[name]

    return get


def test_golden_master_covers_all_cases():
    assert set(META["cases"]) == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden(name, expected_columns, results):
    expected = META["cases"][name]
    status, result = results(name)

    if expected["status"] == "error":
        assert status == "error", f"expected {expected['type']}, got a result"
        assert type(result).__name__ == expected["type"]
        assert str(result) == expected["message"]
        return

    assert status == "ok", f"unexpected {type(result).__name__}: {result}"
    columns = expected_columns(name)
    assert list(result.columns) == list(columns)

    index = expected_index(expected["index"])
    assert result.index.values.dtype == index.dtype
    assert np.array_equal(result.index.values, index)
    assert result.index.freqstr == expected["index"]["freq"]

    for column, values in columns.items():
        np.testing.assert_allclose(
            result[column].values, values, rtol=1e-12, atol=0.0,
            err_msg=f"{name}: {column}",
        )


@pytest.mark.skipif(
    not _BITWISE,
    reason=f"golden master recorded with numpy {_RECORDED['numpy']} on "
           f"{_RECORDED['platform']}/{_RECORDED['machine']}; bitwise comparison "
           "is only meaningful there (force with PYSLPHEAT_GOLDEN_BITWISE=1)",
)
@pytest.mark.parametrize("name", _OK_CASES)
def test_bitwise_identical(name, expected_columns, results):
    status, result = results(name)
    assert status == "ok", f"unexpected {type(result).__name__}: {result}"
    for column, expected in expected_columns(name).items():
        actual = result[column].values
        assert actual.dtype == expected.dtype, f"{name}: {column} dtype"
        assert actual.tobytes() == expected.tobytes(), f"{name}: {column} differs bitwise"

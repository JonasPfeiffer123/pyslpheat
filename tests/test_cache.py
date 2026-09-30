"""
Behaviour of the process-wide caches: results must not depend on whether a
cache is cold or warm, on what callers do with returned data, or on which
thread asks first.
"""

import os
import shutil
import threading

import numpy as np
import pytest

import pyslpheat
from pyslpheat import bdew, vdi4655
from pyslpheat._cache import FileCache

HOLIDAYS = np.array(["2023-01-01", "2023-04-07", "2023-12-25"], dtype="datetime64[D]")


def _bdew(try_path, **kwargs):
    params = dict(annual_heat_kWh=50_000.0, profile_type="GBD", subtype="03",
                  TRY_file_path=try_path, year=2023, dhw_share=0.2)
    params.update(kwargs)
    return bdew.calculate(**params)


def _vdi(try_path, **kwargs):
    params = dict(annual_heating_kWh=15_000.0, annual_dhw_kWh=3_000.0,
                  annual_electricity_kWh=3_500.0, building_type="EFH",
                  number_people_household=2, year=2023, climate_zone="9",
                  TRY=try_path, holidays=HOLIDAYS)
    params.update(kwargs)
    return vdi4655.calculate(**params)


def _assert_frames_identical(actual, expected):
    assert list(actual.columns) == list(expected.columns)
    assert actual.index.equals(expected.index)
    for column in expected.columns:
        assert actual[column].values.tobytes() == expected[column].values.tobytes(), column


def _overwrite(frame):
    """Scribble over everything a caller can reach through a result."""
    for column in frame.columns:
        values = frame[column].values
        if values.flags.writeable:
            values[:] = -12345.0
    index_values = frame.index.values
    if index_values.flags.writeable:
        index_values[:] = index_values[0]


# ── FileCache ────────────────────────────────────────────────────────────────

def _counting_cache(maxsize=8):
    calls = []

    def loader(path):
        calls.append(path)
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    return FileCache(loader, maxsize=maxsize), calls


def test_file_cache_loads_once(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    cache, calls = _counting_cache()
    assert cache.get(str(path)) == "one"
    assert cache.get(str(path)) == "one"
    assert cache.get(path) == "one"                           # PathLike, same file
    assert len(calls) == 1


def test_file_cache_resolves_relative_paths(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    cache, calls = _counting_cache()
    monkeypatch.chdir(tmp_path)
    assert cache.get("a.txt") == "one"
    assert cache.get(str(tmp_path / "a.txt")) == "one"
    assert len(calls) == 1


def test_file_cache_reloads_when_size_changes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    cache, calls = _counting_cache()
    cache.get(str(path))
    path.write_text("three", encoding="utf-8")
    assert cache.get(str(path)) == "three"
    assert len(calls) == 2


def test_file_cache_reloads_when_mtime_changes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    cache, calls = _counting_cache()
    cache.get(str(path))
    stat = os.stat(path)
    path.write_text("two", encoding="utf-8")                  # same size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    assert cache.get(str(path)) == "two"
    assert len(calls) == 2


def test_file_cache_is_bounded(tmp_path):
    cache, calls = _counting_cache(maxsize=2)
    paths = []
    for name in "abc":
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        paths.append(str(path))
        cache.get(str(path))
    assert len(cache._entries) == 2
    cache.get(paths[2])
    assert len(calls) == 3                                    # most recent file still cached
    cache.get(paths[0])
    assert len(calls) == 4                                    # oldest file was evicted


def test_file_cache_missing_file_raises_loader_error(tmp_path):
    cache, _ = _counting_cache()
    with pytest.raises(FileNotFoundError):
        cache.get(str(tmp_path / "missing.txt"))


# ── Weather data ─────────────────────────────────────────────────────────────

def test_import_try_returns_private_copies():
    for module in (bdew, vdi4655):
        first = [a for a in module.import_TRY(pyslpheat.TRY_BAUTZEN_2015) if a is not None]
        expected = [a.copy() for a in first]
        for array in first:
            assert array.flags.writeable
            array[:] = 999.0
        second = [a for a in module.import_TRY(pyslpheat.TRY_BAUTZEN_2015) if a is not None]
        for a, e in zip(second, expected):
            assert np.array_equal(a, e)


def test_changed_try_file_is_read_again(tmp_path):
    path = str(tmp_path / "weather.dat")
    shutil.copyfile(pyslpheat.TRY_BAUTZEN_2015, path)
    before_bdew, before_vdi = _bdew(path), _vdi(path)

    shutil.copyfile(pyslpheat.TRY_BAUTZEN_2045_WINTER, path)
    after_bdew, after_vdi = _bdew(path), _vdi(path)

    _assert_frames_identical(after_bdew, _bdew(pyslpheat.TRY_BAUTZEN_2045_WINTER))
    _assert_frames_identical(after_vdi, _vdi(pyslpheat.TRY_BAUTZEN_2045_WINTER))
    assert not np.array_equal(before_bdew["temperature_C"].values,
                              after_bdew["temperature_C"].values)
    assert not np.array_equal(before_vdi["Q_heat_kWh"].values, after_vdi["Q_heat_kWh"].values)


def test_clear_caches_empties_try_caches():
    _bdew(pyslpheat.TRY_BAUTZEN_2015)
    _vdi(pyslpheat.TRY_BAUTZEN_2015)
    assert bdew._TRY_CACHE._entries and vdi4655._TRY_CACHE._entries
    pyslpheat.clear_caches()
    assert not bdew._TRY_CACHE._entries and not vdi4655._TRY_CACHE._entries


# ── Results are independent of cache state and of what callers do ────────────

CALLS = [
    (_bdew, {}),
    (_bdew, dict(profile_type="GKO", subtype="34", heating_limit_temp=15.0)),
    (_bdew, dict(peak_design_kW=30.0, design_temperature=-14.0)),
    (_bdew, dict(stochastic=True, dhw_draw_events=True)),
    (_vdi, {}),
    (_vdi, dict(building_type="MFH", number_people_household=4)),
]


@pytest.mark.parametrize("func, kwargs", CALLS)
def test_cold_and_warm_calls_agree(func, kwargs):
    pyslpheat.clear_caches()
    cold = func(pyslpheat.TRY_BAUTZEN_2015, **kwargs)
    warm = func(pyslpheat.TRY_BAUTZEN_2015, **kwargs)
    _assert_frames_identical(warm, cold)


@pytest.mark.parametrize("func, kwargs", CALLS)
def test_mutating_a_result_does_not_affect_later_calls(func, kwargs):
    first = func(pyslpheat.TRY_BAUTZEN_2015, **kwargs)
    expected = first.copy(deep=True)
    expected.index = expected.index.copy(deep=True)
    _overwrite(first)
    second = func(pyslpheat.TRY_BAUTZEN_2015, **kwargs)
    _assert_frames_identical(second, expected)


def test_results_are_writeable():
    for frame in (_bdew(pyslpheat.TRY_BAUTZEN_2015), _vdi(pyslpheat.TRY_BAUTZEN_2015)):
        frame.iloc[0, 0] = 1.0
        frame["Q_total_kWh"] *= 2.0


@pytest.mark.parametrize("func", [_bdew, _vdi])
def test_concurrent_first_calls_agree(func):
    expected = func(pyslpheat.TRY_BAUTZEN_2015)
    pyslpheat.clear_caches()
    results, errors = [], []
    barrier = threading.Barrier(4)

    def worker():
        try:
            barrier.wait()
            results.append(func(pyslpheat.TRY_BAUTZEN_2015))
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(results) == 4
    for frame in results:
        _assert_frames_identical(frame, expected)

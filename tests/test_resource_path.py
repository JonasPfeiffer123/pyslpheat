"""
``vdi4655.get_resource_path`` is called with Windows-style legacy paths.
They must resolve under POSIX path rules as well.
"""

import ntpath
import posixpath
import types

import pytest

from pyslpheat import vdi4655

LEGACY_PATHS = {
    "data\\VDI 4655 profiles\\VDI 4655 data\\Faktoren.csv": ("Faktoren.csv",),
    "data\\VDI 4655 profiles\\VDI 4655 load profiles\\EFHWWH.csv": ("load_profiles", "EFHWWH.csv"),
    "data/VDI 4655 profiles/VDI 4655 load profiles/MFHÜSB.csv": ("load_profiles", "MFHÜSB.csv"),
}


@pytest.mark.parametrize("path_module, data_dir", [
    (posixpath, "/opt/pyslpheat/data/vdi4655"),
    (ntpath, "C:\\pyslpheat\\data\\vdi4655"),
])
def test_legacy_paths_resolve(monkeypatch, path_module, data_dir):
    monkeypatch.setattr(vdi4655, "_os", types.SimpleNamespace(sep=path_module.sep, path=path_module))
    monkeypatch.setattr(vdi4655, "_VDI4655_DATA_DIR", data_dir)
    for legacy, parts in LEGACY_PATHS.items():
        assert vdi4655.get_resource_path(legacy) == path_module.join(data_dir, *parts)

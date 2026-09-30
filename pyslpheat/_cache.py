"""
Process-wide caches for weather files and bundled coefficient tables.

``calculate()`` is typically called once per building with the same weather
file and the same bundled tables. Parsing them once per process instead of
once per call removes most of the runtime.

Cached values are shared between calls and must never be modified. Arrays
are therefore stored read-only; functions that hand data to callers return
copies.

See the section *Caching* in ``docs/DOCUMENTATION.md`` for memory use,
invalidation and thread safety.

:author: Dipl.-Ing. (FH) Jonas Pfeiffer
"""

import functools
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, List

_clear_functions: List[Callable[[], None]] = []


def read_only(array):
    """
    Mark a numpy array as read-only and return it.

    :param array: Array owned by a cache
    :type array: np.ndarray
    :return: The same array, no longer writeable
    :rtype: np.ndarray
    """
    array.setflags(write=False)
    return array


def cached_data(loader: Callable) -> Callable:
    """
    Decorator: evaluate *loader* once per distinct argument set and process.

    Intended for bundled package data that never changes at runtime. A call
    that raises is not cached and is retried on the next call.

    :param loader: Function loading the data
    :type loader: Callable
    :return: Cached function
    :rtype: Callable
    """
    cached = functools.lru_cache(maxsize=None)(loader)
    _clear_functions.append(cached.cache_clear)
    return cached


class FileCache:
    """
    Cache parsed file contents per file.

    An entry is valid as long as absolute path, modification time and size
    of the file are unchanged; otherwise the file is parsed again. At most
    *maxsize* files are held, the least recently used one is dropped first.

    :param loader: Function parsing a file path into the value to cache
    :type loader: Callable
    :param maxsize: Maximum number of files held at once
    :type maxsize: int
    """

    def __init__(self, loader: Callable[[Any], Any], maxsize: int = 8) -> None:
        self._loader = loader
        self._maxsize = maxsize
        self._entries: "OrderedDict[Any, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        _clear_functions.append(self.clear)

    def get(self, path: Any) -> Any:
        """
        Return the parsed content of *path*, parsing it only if necessary.

        :param path: File path as accepted by ``open()``
        :return: Value returned by the loader; shared, do not modify
        """
        try:
            key = os.path.abspath(path)
            stat = os.stat(key)
        except (OSError, TypeError, ValueError):
            # Not a file that can be tracked: let the loader raise its usual error
            return self._loader(path)
        stamp = (stat.st_mtime_ns, stat.st_size)

        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] == stamp:
                self._entries.move_to_end(key)
                return entry[1]

        value = self._loader(path)

        with self._lock:
            self._entries[key] = (stamp, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop all entries."""
        with self._lock:
            self._entries.clear()


def clear_caches() -> None:
    """
    Drop all cached weather files and coefficient tables.

    They are loaded again on the next call of ``calculate()``. Not needed in
    normal operation: changed weather files are detected automatically.
    """
    for clear in _clear_functions:
        clear()

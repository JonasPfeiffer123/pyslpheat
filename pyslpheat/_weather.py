"""
Fit hourly weather data to the calendar of the calculation year.

DWD TRY files cover 8760 hours, i.e. 365 days. For a leap year the weather of
28 February is repeated for 29 February; weekdays and holidays are taken from
the real calendar of the year elsewhere.

:author: Dipl.-Ing. (FH) Jonas Pfeiffer
"""

import calendar
from typing import Tuple

import numpy as np

_HOURS_PER_DAY = 24
_DAYS_BEFORE_FEB_29 = 31 + 28


def fit_to_year(year: int, *hourly: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Return hourly weather arrays that cover every hour of *year*.

    Arrays that already match the year are returned unchanged. For a leap year
    and arrays of 365 days, the 24 hours of 28 February are inserted again as
    29 February.

    :param year: Calculation year
    :type year: int
    :param hourly: Hourly arrays of equal length from one TRY file
    :type hourly: np.ndarray
    :return: The arrays, each with 8760 or 8784 values
    :rtype: Tuple[np.ndarray, ...]
    :raises ValueError: If the number of hours fits neither the year nor, for a
        leap year, 365 days
    """
    hours = len(hourly[0])
    leap = calendar.isleap(year)
    needed = (366 if leap else 365) * _HOURS_PER_DAY
    if hours == needed:
        return hourly

    if leap and hours == 365 * _HOURS_PER_DAY:
        mar1 = _DAYS_BEFORE_FEB_29 * _HOURS_PER_DAY
        feb28 = mar1 - _HOURS_PER_DAY
        return tuple(np.concatenate([a[:mar1], a[feb28:mar1], a[mar1:]]) for a in hourly)

    if leap:
        raise ValueError(
            f"TRY file has {hours} hourly values; the leap year {year} needs 8784, "
            "or 8760 to repeat 28 February for 29 February")
    raise ValueError(f"TRY file has {hours} hourly values; the year {year} needs 8760")

import datetime
import sqlite3
import sys
from functools import cache

import numpy as np
import pandas as pd


@cache
def sqlite_date_adapters() -> None:
    # The following code has been copied here from in Python 3.11:
    # `sqlite3.dbapi2.register_adapters_and_converters`
    # Including minor modifications to source code to pass linting
    # https://docs.python.org/3/license.html#psf-license

    def adapt_date(val):
        return val.isoformat()

    def adapt_datetime(val):
        return val.isoformat(" ")

    def adapt_time(t) -> str:
        return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}"

    def adapt_np_datetime64(val):
        return np.datetime_as_string(val, unit="us").replace("T", " ")

    def convert_date(val):
        return datetime.date(*map(int, val.split(b"-")))

    def convert_timestamp(val):
        datepart, timepart = val.split(b" ")
        year, month, day = map(int, datepart.split(b"-"))
        timepart_full = timepart.split(b".")
        hours, minutes, seconds = map(int, timepart_full[0].split(b":"))
        microseconds = int(f"{timepart_full[1].decode():0<6.6}") if len(timepart_full) == 2 else 0

        val = datetime.datetime(year, month, day, hours, minutes, seconds, microseconds)
        return val

    if sys.version_info >= (3, 12):
        # Python 3.12 has removed datetime support from sqlite3
        # https://github.com/python/cpython/pull/93095
        sqlite3.register_adapter(datetime.date, adapt_date)
        sqlite3.register_adapter(datetime.datetime, adapt_datetime)
        sqlite3.register_converter("date", convert_date)
        sqlite3.register_converter("timestamp", convert_timestamp)

    sqlite3.register_adapter(datetime.time, adapt_time)
    sqlite3.register_adapter(np.datetime64, adapt_np_datetime64)
    sqlite3.register_adapter(pd.Timestamp, adapt_datetime)

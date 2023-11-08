import datetime
import sqlite3
import sys
from functools import cache


@cache
def sqlite_date_adapters() -> None:
    # Python 3.12 has removed datetime support from sqlite3
    # https://github.com/python/cpython/pull/93095
    # The following code has been copied here from in Python 3.11:
    # `sqlite3.dbapi2.register_adapters_and_converters`
    # Including minor modifications to source code to pass linting
    # https://docs.python.org/3/license.html#psf-license

    if sys.version_info < (3, 12):
        return

    def adapt_date(val):
        return val.isoformat()

    def adapt_datetime(val):
        return val.isoformat(" ")

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

    sqlite3.register_adapter(datetime.date, adapt_date)
    sqlite3.register_adapter(datetime.datetime, adapt_datetime)
    sqlite3.register_converter("date", convert_date)
    sqlite3.register_converter("timestamp", convert_timestamp)

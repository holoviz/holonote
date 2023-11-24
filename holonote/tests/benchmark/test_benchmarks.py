import numpy as np
import pandas as pd
import pytest

from holonote.annotate import Annotator, SQLiteDB

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("items", [10, 100, 1_000])
def test_define_annotations(benchmark, items):
    rng = np.random.default_rng(1337)
    start_time = np.arange(0, items * 2, 2)
    end_time = start_time + 1
    data = pd.DataFrame(
        {
            "start_time": start_time,
            "end_time": end_time,
            "description": rng.choice(["A", "B"], items),
        }
    )
    annotator = Annotator({"TIME": int}, connector=SQLiteDB(filename=":memory:"))
    annotator.define_annotations(data, TIME=("start_time", "end_time"))

    @benchmark
    def bench():
        annotator.define_annotations(data, TIME=("start_time", "end_time"))

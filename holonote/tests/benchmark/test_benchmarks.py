import numpy as np
import pandas as pd
import pytest

from holonote.annotate import Annotator, SQLiteDB

pytestmark = pytest.mark.benchmark


def setup_annotator(items) -> tuple[Annotator, pd.DataFrame]:
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
    return annotator, data


@pytest.mark.parametrize("items", [10, 100, 1000])
def test_define_annotations(benchmark, items) -> None:
    annotator, data = setup_annotator(items)

    @benchmark
    def bench() -> None:
        annotator.define_annotations(data, TIME=("start_time", "end_time"))


@pytest.mark.parametrize("items", [10, 100, 1000])
def test_commit(benchmark, items) -> None:
    annotator, data = setup_annotator(items)
    annotator.define_annotations(data, TIME=("start_time", "end_time"))

    @benchmark
    def bench() -> None:
        annotator.commit()

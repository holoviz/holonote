# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
import pandas as pd

from holonote.annotate import Annotator, SQLiteDB


class DefineSuite:
    params = [10, 100, 1_000, 10_000]
    param_names = ["items"]

    def setup(self, items) -> None:
        rng = np.random.default_rng(1337)
        start_time = np.arange(0, items, 2)
        end_time = start_time + 1
        self.data = pd.DataFrame(
            {
                "start_time": start_time,
                "end_time": end_time,
                "description": rng.choice(["A", "B"], items // 2),
            }
        )
        self.annotator = Annotator({"TIME": int}, connector=SQLiteDB(filename=":memory:"))

    def time_define_annotations(self, items) -> None:
        self.annotator.define_annotations(self.data, TIME=("start_time", "end_time"))

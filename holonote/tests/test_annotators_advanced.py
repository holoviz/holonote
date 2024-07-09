from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from holonote.annotate import Annotator, SQLiteDB


@pytest.mark.skip("Not supported having multiple region types for same dimension")
def test_multipoint_range_commit_insertion(multiple_region_annotator):
    descriptions = ["A point insertion", "A range insertion"]
    timestamp = np.datetime64("2022-06-06")
    multiple_region_annotator.set_point(timestamp)
    multiple_region_annotator.add_annotation(description=descriptions[0])

    start, end = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
    multiple_region_annotator.set_range(start, end)
    multiple_region_annotator.add_annotation(description=descriptions[1])

    multiple_region_annotator.commit(return_commits=True)

    # FIXME! Index order is inverted?
    df = pd.DataFrame(
        {
            "uuid": pd.Series(multiple_region_annotator.df.index[::-1], dtype=object),
            "point_TIME": [timestamp, pd.NaT],
            "start_TIME": [pd.NaT, start],
            "end_TIME": [pd.NaT, end],
            "description": descriptions,
        }
    ).set_index("uuid")

    sql_df = multiple_region_annotator.connector.load_dataframe()
    pd.testing.assert_frame_equal(sql_df, df)


def test_infer_kdim_dtype_image():
    xvals, yvals = np.linspace(-4, 0, 202), np.linspace(4, 0, 202)
    xs, ys = np.meshgrid(xvals, yvals)
    image = hv.Image(np.sin(ys * xs), kdims=["A", "B"])
    assert Annotator._infer_kdim_dtypes(image) == {"A": np.float64, "B": np.float64}


def test_infer_kdim_dtype_curve():
    curve = hv.Curve(
        (np.arange("2005-02", "2005-03", dtype="datetime64[D]"), range(28)), kdims=["TIME"]
    )
    assert Annotator._infer_kdim_dtypes(curve) == {"TIME": np.datetime64}


def test_multiplot_add_annotation(multiple_annotators):
    start_time, end_time = np.datetime64("2005-02-13"), np.datetime64("2005-02-16")
    multiple_annotators.set_regions(TIME=(start_time, end_time))
    multiple_annotators.set_regions(x=(-0.25, 0.25), y=(-0.1, 0.1))
    multiple_annotators.add_annotation(description="Multi-plot annotation", uuid="A")
    multiple_annotators.commit()

    d = {
        "start[TIME]": {"A": start_time},
        "end[TIME]": {"A": end_time},
        "start[x]": {"A": -0.25},
        "end[x]": {"A": 0.25},
        "start[y]": {"A": -0.1},
        "end[y]": {"A": 0.1},
        "description": {"A": "Multi-plot annotation"},
    }
    expected = pd.DataFrame(d)
    expected.index.name = "uuid"
    pd.testing.assert_frame_equal(multiple_annotators.df, expected)


class TestAnnotatorMultipleStringFields:
    def test_insertion_values(self, multiple_fields_annotator):
        start, end = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        multiple_fields_annotator.set_regions(TIME=(start, end))
        multiple_fields_annotator.add_annotation(
            field1="A test field", field2="Another test field"
        )
        commits = multiple_fields_annotator.commit(return_commits=True)
        kwargs = commits[0]["kwargs"]
        assert len(commits) == 1, "Only one insertion commit made"
        assert "uuid" in kwargs, "Expected uuid primary key in kwargs"
        kwargs.pop("uuid")
        assert kwargs == {
            "field1": "A test field",
            "field2": "Another test field",
            "start_TIME": start,
            "end_TIME": end,
        }

    def test_commit_insertion(self, multiple_fields_annotator):
        start, end = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        field1 = "A test field"
        field2 = "Another test field"
        multiple_fields_annotator.set_regions(TIME=(start, end))
        multiple_fields_annotator.add_annotation(field1=field1, field2=field2)
        multiple_fields_annotator.commit(return_commits=True)

        df = pd.DataFrame(
            {
                "uuid": pd.Series(multiple_fields_annotator.df.index[0], dtype=object),
                "start_TIME": [start],
                "end_TIME": [end],
                "field1": [field1],
                "field2": [field2],
            }
        ).set_index("uuid")

        sql_df = multiple_fields_annotator.connector.load_dataframe()
        pd.testing.assert_frame_equal(sql_df, df)

    def test_commit_update_set_range(self, multiple_fields_annotator):
        start1, end1 = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        start2, end2 = np.datetime64("2023-06-06"), np.datetime64("2023-06-08")
        multiple_fields_annotator.set_regions(TIME=(start1, end1))
        multiple_fields_annotator.add_annotation(field1="Field 1.1", field2="Field 1.2")
        multiple_fields_annotator.set_regions(TIME=(start2, end2))
        multiple_fields_annotator.add_annotation(field1="Field 2.1", field2="Field 2.2")
        multiple_fields_annotator.commit(return_commits=True)
        multiple_fields_annotator.update_annotation_fields(
            multiple_fields_annotator.df.index[0], field1="NEW Field 1.1"
        )
        multiple_fields_annotator.commit(return_commits=True)
        sql_df = multiple_fields_annotator.connector.load_dataframe()
        assert set(sql_df["field1"]) == {"NEW Field 1.1", "Field 2.1"}

    def test_commit_update_set_region(self, multiple_fields_annotator):
        start1, end1 = np.datetime64("2022-06-06"), np.datetime64("2022-06-08")
        start2, end2 = np.datetime64("2023-06-06"), np.datetime64("2023-06-08")
        multiple_fields_annotator.set_regions(TIME=(start1, end1))
        multiple_fields_annotator.add_annotation(field1="Field 1.1", field2="Field 1.2")
        multiple_fields_annotator.set_regions(TIME=(start2, end2))
        multiple_fields_annotator.add_annotation(field1="Field 2.1", field2="Field 2.2")
        multiple_fields_annotator.commit(return_commits=True)
        multiple_fields_annotator.update_annotation_fields(
            multiple_fields_annotator.df.index[0], field1="NEW Field 1.1"
        )
        multiple_fields_annotator.commit(return_commits=True)
        sql_df = multiple_fields_annotator.connector.load_dataframe()
        assert set(sql_df["field1"]) == {"NEW Field 1.1", "Field 2.1"}


@pytest.mark.parametrize("method", ["new", "same"])
def test_reconnect(method, tmp_path):
    db_path = str(tmp_path / "test.db")

    if method == "new":
        conn1 = SQLiteDB(filename=db_path)
        conn2 = SQLiteDB(filename=db_path)
    elif method == "same":
        conn1 = conn2 = SQLiteDB(filename=db_path)

    # Create annotator with data and commit
    a1 = Annotator(
        spec={"TIME": np.datetime64},
        fields=["description"],
        connector=conn1,
    )
    times = pd.date_range("2022-06-09", "2022-06-13")
    for t1, t2 in zip(times[:-1], times[1:]):
        a1.set_regions(TIME=(t1, t2))
        a1.add_annotation(description="A programmatically defined annotation")
    a1.commit(return_commits=True)

    # Save internal dataframes
    a1_df = a1.df.copy()
    a1_region = a1.annotation_table._region_df.copy()
    a1_field = a1.annotation_table._field_df.copy()

    # Add new connector
    a2 = Annotator(
        spec={"TIME": np.datetime64},
        fields=["description"],
        connector=conn2,
    )
    a2_df = a2.df.copy()
    a2_region = a2.annotation_table._region_df.copy()
    a2_field = a2.annotation_table._field_df.copy()

    pd.testing.assert_frame_equal(a1_df, a2_df)
    pd.testing.assert_frame_equal(a1_region, a2_region)
    pd.testing.assert_frame_equal(a1_field, a2_field)


def test_define_annotations_multiple_without_without_all_fields(conn_sqlite_uuid) -> None:
    annotator = Annotator(
        {"TIME": float}, fields=["description", "category"], connector=conn_sqlite_uuid
    )

    # Sample data without description field
    data = {
        "category": ["A", "B", "A", "C", "B"],
        "start_number": [1, 6, 11, 16, 21],
        "end_number": [5, 10, 15, 20, 25],
    }

    annotator.define_annotations(pd.DataFrame(data), TIME=("start_number", "end_number"))

    # Add description field
    data["description"] = (None,) * 5

    annotator.define_annotations(pd.DataFrame(data), TIME=("start_number", "end_number"))


def test_define_annotations_with_different_field(conn_sqlite_uuid) -> None:
    annotator = Annotator({"TIME": float}, fields=["description"], connector=conn_sqlite_uuid)
    data = {
        "category": ["A", "B", "A", "C", "B"],
        "start_number": [1, 6, 11, 16, 21],
        "end_number": [5, 10, 15, 20, 25],
    }
    annotator.define_annotations(
        pd.DataFrame(data), TIME=("start_number", "end_number"), description="category"
    )
    assert (annotator.df["description"] == data["category"]).all()


def test_define_annotations_with_multiple_field_warn(conn_sqlite_uuid) -> None:
    annotator = Annotator(
        {"TIME": float}, fields=["description", "category"], connector=conn_sqlite_uuid
    )
    data = {
        "category": ["A", "B", "A", "C", "B"],
        "start_number": [1, 6, 11, 16, 21],
        "end_number": [5, 10, 15, 20, 25],
    }

    # Description set to use category will give a warning,
    # which raises an error in the testsuite
    msg = "Input 'category' has overlapping name with a field or spec"
    with pytest.raises(UserWarning, match=msg):
        annotator.define_annotations(
            pd.DataFrame(data), TIME=("start_number", "end_number"), description="category"
        )


def test_define_annotations_clear_region(conn_sqlite_uuid) -> None:
    # See: https://github.com/holoviz/holonote/pull/50
    annotator = Annotator({"x": float}, connector=conn_sqlite_uuid)
    data = {"description": ["A", "B"], "start_number": [0.1, 0.5], "end_number": [0.4, 0.8]}
    annotator.define_annotations(pd.DataFrame(data), x=("start_number", "end_number"))
    assert annotator.region == {}
    assert annotator._last_region == {}


def test_update_region(multiple_annotators, conn_sqlite_uuid) -> None:
    annotator = multiple_annotators

    # Add one annotation and commit
    first = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    annotator.set_regions(TIME=first)
    annotator.add_annotation(description="test")
    annotator.commit()

    # In the Annotation Table
    df = annotator.df
    assert df.shape[0] == 1
    assert (df["start[TIME]"] == first[0]).all()
    assert (df["end[TIME]"] == first[1]).all()

    # In the database
    df = conn_sqlite_uuid.load_dataframe()
    assert df.shape[0] == 1
    assert (df["start_TIME"] == first[0]).all()
    assert (df["end_TIME"] == first[1]).all()

    # Update annotation region and commit
    second = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")
    annotator.set_regions(TIME=second)
    annotator.update_annotation_region(df.index[0])
    annotator.commit()

    # In the Annotation Table
    df = annotator.df
    assert df.shape[0] == 1
    assert (df["start[TIME]"] == second[0]).all()
    assert (df["end[TIME]"] == second[1]).all()

    # In the database
    df = conn_sqlite_uuid.load_dataframe()
    assert df.shape[0] == 1
    assert (df["start_TIME"] == second[0]).all()
    assert (df["end_TIME"] == second[1]).all()


class TestEvent:
    @pytest.fixture(autouse=True)
    def _setup_count(self, multiple_annotators):
        self.count = {"create": 0, "update": 0, "delete": 0, "commit": 0}

        def count(event) -> None:
            self.count[event.type] += 1

        def commit_count(event) -> None:
            self.count["commit"] += 1

        multiple_annotators.on_event(count)
        multiple_annotators.on_commit(commit_count)

    def check(self, create=0, update=0, delete=0, commit=0):
        assert self.count["create"] == create
        assert self.count["update"] == update
        assert self.count["delete"] == delete
        assert self.count["commit"] == commit

    def test_create(self, multiple_annotators):
        annotator = multiple_annotators
        annotator.set_regions(x=(0, 1))
        self.check(create=0, update=0, delete=0)
        annotator.add_annotation(description="test")
        self.check(create=1, update=0, delete=0)
        annotator.commit()
        self.check(create=1, update=0, delete=0, commit=1)
        annotator.commit()  # empty, no change
        self.check(create=1, update=0, delete=0, commit=1)

    def test_update_fields(self, multiple_annotators):
        annotator = multiple_annotators
        annotator.set_regions(x=(0, 1))
        self.check(create=0, update=0, delete=0)
        annotator.add_annotation(description="test")
        self.check(create=1, update=0, delete=0)

        index = annotator.df.index[0]
        annotator.update_annotation_fields(index, description="updated")
        self.check(create=1, update=1, delete=0)

    def test_update_regions(self, multiple_annotators):
        annotator = multiple_annotators
        annotator.set_regions(x=(0, 1))
        self.check(create=0, update=0, delete=0)
        annotator.add_annotation(description="test")
        self.check(create=1, update=0, delete=0)

        index = annotator.df.index[0]
        annotator.set_regions(x=(1, 2))
        annotator.update_annotation_region(index)
        self.check(create=1, update=1, delete=0)

    def test_delete(self, multiple_annotators):
        annotator = multiple_annotators
        annotator.set_regions(x=(0, 1))
        self.check(create=0, update=0, delete=0)
        annotator.add_annotation(description="test")
        self.check(create=1, update=0, delete=0)

        index = annotator.df.index[0]
        annotator.delete_annotation(index)
        self.check(create=1, update=0, delete=1)

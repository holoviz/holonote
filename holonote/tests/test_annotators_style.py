import holoviews as hv
import pandas as pd
import pytest

from holonote.annotate import Annotator, Style
from holonote.tests.util import get_editor, get_indicator


@pytest.fixture()
def cat_annotator(conn_sqlite_uuid) -> Annotator:
    # Initialize annotator
    annotator = Annotator(
        {"x": float},
        fields=["description", "category"],
        connector=conn_sqlite_uuid,
        groupby="category",
    )
    # Add data to annotator
    data = {
        "category": ["A", "B", "A", "C", "B"],
        "start_number": [1, 6, 11, 16, 21],
        "end_number": [5, 10, 15, 20, 25],
        "description": list("ABCDE"),
    }
    annotator.define_annotations(pd.DataFrame(data), x=("start_number", "end_number"))
    # Setup display
    annotator.get_display("x")
    return annotator


@pytest.fixture()
def cat_element() -> hv.Curve:
    return hv.Curve(range(30), kdims=["x"])


def test_style_accessor(cat_annotator) -> None:
    assert isinstance(cat_annotator.style, Style)

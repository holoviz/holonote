from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn

from holonote.app import AnnotatorTable, PanelWidgets


def test_panel_app(annotator_range1d):
    w = PanelWidgets(annotator_range1d)
    assert isinstance(w.fields_widgets, pn.Column)
    assert isinstance(w.tool_widgets, pn.Row)


def test_as_popup(annotator_range1d):
    w = PanelWidgets(annotator_range1d, as_popup=True)
    assert not w._layout.visible
    for display in w.annotator._displays.values():
        assert display._edit_streams[0].popup
        assert display._tap_stream.popup
    assert w.__panel__().visible


def test_tabulator(annotator_range1d):
    t = AnnotatorTable(annotator_range1d)
    assert isinstance(t.tabulator, pn.widgets.Tabulator)

    annotator_range1d.set_regions(TIME=(np.datetime64("2022-06-06"), np.datetime64("2022-06-08")))
    annotator_range1d.add_annotation(description="A test annotation!")
    assert len(t.tabulator.value) == 1
    assert t.tabulator.value.iloc[0, 0] == pd.Timestamp("2022-06-06")
    assert t.tabulator.value.iloc[0, 1] == pd.Timestamp("2022-06-08")
    assert t.tabulator.value.iloc[0, 2] == "A test annotation!"
    assert "darkgray" in t.tabulator.style.to_html()

    annotator_range1d.commit(return_commits=True)
    assert "darkgray" not in t.tabulator.style.to_html()

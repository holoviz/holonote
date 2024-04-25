from __future__ import annotations

import panel as pn

from holonote.app import PanelWidgets


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
import panel as pn

from holonote.app import PanelWidgets


def test_panel_app(annotator_range1d):
    w = PanelWidgets(annotator_range1d)
    assert isinstance(w.fields_widgets, pn.Column)
    assert isinstance(w.tool_widgets, pn.Row)

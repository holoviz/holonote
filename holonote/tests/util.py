import holoviews as hv

bk_renderer = hv.renderer("bokeh")


def _get_display(annotator, kdims=None) -> hv.Element:
    if kdims is None:
        kdims = next(iter(annotator._displays))
    kdims = (kdims,) if isinstance(kdims, str) else tuple(kdims)
    disp = annotator.get_display(*kdims)
    bk_renderer.get_plot(disp.element)  # Trigger rendering
    return disp


def get_editor(annotator, element_type, kdims=None):
    el = _get_display(annotator, kdims).region_editor()
    for e in el.last.traverse():
        if isinstance(e, element_type):
            return e


def get_indicator(annotator, element_type, kdims=None):
    si = _get_display(annotator, kdims).static_indicators
    return next(iter(si.data.values()))

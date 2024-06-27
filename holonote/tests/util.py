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


def get_editor_data(annotator, element_type, kdims=None):
    el = get_editor(annotator, element_type, kdims)
    return getattr(el, "data", None)


def get_indicator(annotator, element_type, kdims=None):
    si = _get_display(annotator, kdims).indicators().last
    yield from si.data.values()


def get_indicator_data(annotator, element_type, kdims=None):
    for el in get_indicator(annotator, element_type, kdims):
        yield el.data


def get_display_data_from_plot(plot, element_type, kdims):
    display_data = None

    for el in plot.traverse():
        if isinstance(el, element_type):
            display_data = getattr(el, "data", None)
            break

    if display_data is not None:
        cols = [f"start[{dim}]" for dim in kdims] + [f"end[{dim}]" for dim in kdims]
        display_data = display_data.iloc[0][cols].to_list()

    return display_data
